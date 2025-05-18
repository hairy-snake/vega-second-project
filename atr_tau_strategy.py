from dataclasses import dataclass
from typing import List

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity)
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity


@dataclass
class ATRTauResetParams(BaseStrategyParams):
    """
    Parameters for the τ-reset strategy:
    - TAU: The width of the price range (bucket) around the current price.
    - INITIAL_BALANCE: The initial balance for liquidity allocation.
    """
    TAU: float
    INITIAL_BALANCE: float
    ATR_PERIOD: int = 14
    TAU_SCALING_FACTOR: float = 1.5
    MIN_TAU: float = 8
    MAX_TAU: float = 15.0


class ATRTauResetStrategy(BaseStrategy):
    """
    The τ-reset strategy manages liquidity in Uniswap v3 by concentrating it
    within a price range around the current market price. If the price exits this range,
    the liquidity is reallocated. If no position is open, it deposits funds first.

    Based on
    https://drops.dagstuhl.de/storage/00lipics/lipics-vol282-aft2023/LIPIcs.AFT.2023.25/LIPIcs.AFT.2023.25.pdf
    """

    # Decimals for token0 and token1 for Uniswap V3 LP Config
    # This is pool-specific and should be set before running the strategy
    # They are not in the BaseStrategyParams because they are not hyperparameters
    # and are specific to the pool being traded consts.
    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(self, params: ATRTauResetParams, debug: bool = False, *args, **kwargs):
        self._params: ATRTauResetParams = None  # set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False
        self.price_history: List[float] = []
        self.atr: Optional[float] = None

    def set_up(self):
        """
        Register the Uniswap V3 LP entity to manage liquidity in the pool.
        """
        self.register_entity(NamedEntity(
            entity_name='UNISWAP_V3',
            entity=UniswapV3LPEntity(
                UniswapV3LPConfig(
                    token0_decimals=self.token0_decimals,
                    token1_decimals=self.token1_decimals
                )
            )
        ))
        assert isinstance(self.get_entity('UNISWAP_V3'), UniswapV3LPEntity)

    def _update_atr(self, current_price: float) -> None:
        """
        Update the Average True Range (ATR) indicator based on price history.
        """
        self.price_history.append(current_price)
        
        if len(self.price_history) > self._params.ATR_PERIOD * 2:
            self.price_history = self.price_history[-self._params.ATR_PERIOD * 2:]
        
        if len(self.price_history) < self._params.ATR_PERIOD + 1:
            return
        
        true_ranges = []
        for i in range(1, len(self.price_history)):
            high = max(self.price_history[i], self.price_history[i-1])
            low = min(self.price_history[i], self.price_history[i-1])
            true_ranges.append(high - low)
        
        if len(true_ranges) >= self._params.ATR_PERIOD:
            atr_period = min(self._params.ATR_PERIOD, len(true_ranges))
            current_atr = sum(true_ranges[-atr_period:]) / atr_period
            
            if self.atr is None:
                self.atr = current_atr
            else:
                self.atr = (self.atr * (atr_period - 1) + current_atr) / atr_period

    def _get_adjusted_tau(self, current_price: float) -> float:
        """
        Adjust tau based on current volatility (ATR).
        Returns the adjusted tau value, clamped between MIN_TAU and MAX_TAU.
        """
        if self.atr is None or current_price == 0:
            return self._params.TAU
        
        volatility = self.atr / current_price
        
        adjusted_tau = self._params.TAU * self._params.TAU_SCALING_FACTOR
        self._debug(f"High volatility detected ({volatility:.2%}), adjusting tau from {self._params.TAU} to {adjusted_tau}")

        
        return max(self._params.MIN_TAU, min(self._params.MAX_TAU, adjusted_tau))


    def predict(self) -> List[ActionToTake]:
        """
        Main logic of the strategy. Checks if the price has moved outside
        the predefined range and takes actions if necessary.
        """
        # Retrieve the pool state from the registered entity
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity.global_state
        current_price = global_state.price  # Get the current market price

        self._update_atr(current_price)
        # Check if we need to deposit funds into the LP before proceeding
        if not uniswap_entity.is_position and not self.deposited_initial_funds:
            self._debug("No active position. Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()

        if not uniswap_entity.is_position:
            self._debug("No active position. Run first rebalance")
            return self._rebalance()

        # Calculate the boundaries of the price range (bucket)
        lower_bound, upper_bound = uniswap_entity.internal_state.price_lower, uniswap_entity.internal_state.price_upper

        # If the price moves outside the range, reallocate liquidity
        if current_price < lower_bound or current_price > upper_bound:
            self._debug(f"Rebalance {current_price} moved outside range [{lower_bound}, {upper_bound}].")
            return self._rebalance()
        return []

    def _deposit_to_lp(self) -> List[ActionToTake]:
        """
        Deposit funds into the Uniswap LP if no position is currently open.
        """
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]

    def _rebalance(self) -> List[ActionToTake]:
        """
        Reallocate liquidity to a new range centered around the new price.
        """
        actions = []
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')

        # Step 1: Withdraw liquidity from the current range
        if entity.internal_state.liquidity > 0:
            actions.append(
                ActionToTake(entity_name='UNISWAP_V3', action=Action(action='close_position', args={}))
            )
            self._debug("Liquidity withdrawn from the current range.")

        # Step 2: Calculate new range boundaries
        # tau = self._params.TAU
        reference_price: float = entity.global_state.price
        adjusted_tau = self._get_adjusted_tau(reference_price)
        tick_spacing = self.tick_spacing
        price_lower = reference_price * 1.0001 ** (-adjusted_tau * tick_spacing)
        price_upper = reference_price * 1.0001 ** (adjusted_tau * tick_spacing)

        # Step 3: Open a new position centered around the new price
        delegate_get_cash = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash
        actions.append(ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(
                action='open_position',
                args={
                    'amount_in_notional': delegate_get_cash,  # Allocate all available cash
                    'price_lower': price_lower,
                    'price_upper': price_upper
                }
            )
        ))
        self._debug(f"New position opened with range [{price_lower}, {price_upper}]. using tau {adjusted_tau}")
        return actions