from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity)
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity


@dataclass
class TimedTauResetParams(BaseStrategyParams):
    """
    Enhanced parameters for the τ-reset strategy with EMA-based rebalancing:
    - TAU: The width of the price range (bucket) around the current price.
    - INITIAL_BALANCE: The initial balance for liquidity allocation.
    - EMA_PERIOD: Period for exponential moving average calculation.
    - REBALANCE_DELTA: Number of iterations between forced rebalances.
    - EMERGENCY_REBALANCE_THRESHOLD: Multiplier for emergency rebalance when price moves too far.
    """
    TAU: float
    INITIAL_BALANCE: float
    EMA_PERIOD: int = 20
    REBALANCE_DELTA: int = 100
    EMERGENCY_REBALANCE_THRESHOLD: float = 1.5


class TimedTauResetStrategy(BaseStrategy):
    """
    Enhanced τ-reset strategy with:
    1. EMA-based price tracking instead of raw price
    2. Periodic rebalancing every delta iterations
    3. Emergency rebalancing when price moves too far from EMA
    4. Custom volatility indicator to adjust position sizing
    """

    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(self, params: TimedTauResetParams, debug: bool = False, *args, **kwargs):
        self._params: TimedTauResetParams = None  # set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False
        self.iteration_count = 0
        self.price_history = []
        self.ema: Optional[float] = None
        self.volatility_indicator: Optional[float] = None

    def set_up(self):
        """Register the Uniswap V3 LP entity and initialize indicators."""
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

    def _update_indicators(self, current_price: float):
        """Update EMA and custom volatility indicator."""
        self.price_history.append(current_price)
        
        if len(self.price_history) == 1:
            self.ema = current_price
        else:
            alpha = 2 / (self._params.EMA_PERIOD + 1)
            self.ema = alpha * current_price + (1 - alpha) * self.ema
        
        if len(self.price_history) >= 10:  # Need at least 10 points for meaningful volatility
            returns = np.diff(np.log(self.price_history[-10:]))
            self.volatility_indicator = np.std(returns)
        else:
            self.volatility_indicator = None

    def predict(self) -> List[ActionToTake]:
        """Main strategy logic with enhanced rebalancing conditions."""
        self.iteration_count += 1
        
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity.global_state
        current_price = global_state.price
        
        self._update_indicators(current_price)
        
        if not uniswap_entity.is_position and not self.deposited_initial_funds:
            self._debug("No active position. Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()

        should_rebalance = False
        rebalance_reason = ""
        
        if not uniswap_entity.is_position:
            should_rebalance = True
            rebalance_reason = "no active position"
        
        elif self.iteration_count % self._params.REBALANCE_DELTA == 0:
            should_rebalance = True
            rebalance_reason = f"periodic rebalance (every {self._params.REBALANCE_DELTA} iterations)"
        
        elif self.ema is not None:
            price_deviation = abs(current_price - self.ema) / self.ema
            if price_deviation > (self._params.EMERGENCY_REBALANCE_THRESHOLD - 1):
                should_rebalance = True
                rebalance_reason = f"emergency rebalance (price deviation {price_deviation:.2%} > threshold)"
        
        if should_rebalance:
            self._debug(f"Rebalancing triggered: {rebalance_reason}")
            return self._rebalance()
        
        return []

    def _deposit_to_lp(self) -> List[ActionToTake]:
        """Deposit funds into the Uniswap LP."""
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]

    def _rebalance(self) -> List[ActionToTake]:
        """Enhanced rebalance logic with volatility-adjusted position sizing."""
        actions = []
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')

        if entity.is_position and entity.internal_state.liquidity > 0:
            actions.append(ActionToTake(entity_name='UNISWAP_V3', action=Action(action='close_position', args={})))
            self._debug("Closed existing position.")

        # Use current price as reference to ensure range includes it
        reference_price = entity.global_state.price
        tau = self._params.TAU
        
        if self.volatility_indicator is not None:
            volatility_adjustment = 1 / (1 + self.volatility_indicator * 10)  # Empirical scaling
            tau *= volatility_adjustment
            self._debug(f"Volatility-adjusted tau: {tau:.4f} (raw: {self._params.TAU})")

        tick_spacing = self.tick_spacing
        price_lower = reference_price * 1.0001 ** (-tau * tick_spacing)
        price_upper = reference_price * 1.0001 ** (tau * tick_spacing)

        delegate_get_cash = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash
        actions.append(ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(
                action='open_position',
                args={
                    'amount_in_notional': delegate_get_cash,
                    'price_lower': price_lower,
                    'price_upper': price_upper
                }
            )
        ))
        self._debug(f"Opened new position around {reference_price:.4f} with range [{price_lower:.4f}, {price_upper:.4f}]")
        return actions