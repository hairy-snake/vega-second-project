from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
from collections import deque

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity, Observation)
from fractal.core.entities import UniswapV3LPGlobalState, UniswapV3LPEntity, UniswapV3LPConfig
from fractal.core.base import GlobalState

@dataclass
class CorrelationTauParams(BaseStrategyParams):
    """
    Parameters for the correlation-based τ-reset strategy:
    - BASE_TAU: The base width of the price range
    - INITIAL_BALANCE: The initial balance for liquidity allocation
    - CORRELATION_WINDOW: Number of observations for correlation calculation
    - MIN_TAU: Minimum tau value for position width
    - MAX_TAU: Maximum tau value for position width
    - CORRELATION_THRESHOLD: Minimum correlation to consider meaningful
    - HIGH_CORR_SCALE: Scale factor for tau when correlation is high
    - LOW_CORR_SCALE: Scale factor for tau when correlation is low
    """
    BASE_TAU: float
    INITIAL_BALANCE: float
    CORRELATION_WINDOW: int = 168  # 1 week of hourly data
    MIN_TAU: float = 5.0
    MAX_TAU: float = 20.0
    CORRELATION_THRESHOLD: float = 0.7
    HIGH_CORR_SCALE: float = 0.7  # Reduce tau for high correlation
    LOW_CORR_SCALE: float = 1.3   # Increase tau for low correlation


class CorrelationTauStrategy(BaseStrategy):
    """
    A τ-reset strategy that adjusts position ranges based on correlations with external instruments.
    
    When price movements are highly correlated with external reference instruments,
    the strategy assumes more predictable behavior and uses narrower ranges (smaller tau).
    When correlation is low, it uses wider ranges to capture unpredictable movements.
    """

    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(self, params: CorrelationTauParams, debug: bool = False, *args, **kwargs):
        self._params: CorrelationTauParams = None  # set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False
        
        self.uniswap_prices = deque(maxlen=self._params.CORRELATION_WINDOW)
        self.external_prices = deque(maxlen=self._params.CORRELATION_WINDOW)
        
        self.current_correlation: Optional[float] = None
        self.adjusted_tau: float = self._params.BASE_TAU

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

    def _update_price_data(self, observation: Observation):
        """Update price history from observation"""
        uniswap_entity = self.get_entity('UNISWAP_V3')
        uniswap_price = uniswap_entity.global_state.price
        self.uniswap_prices.append(uniswap_price)
        
        external_price = None
        for entity_name in ['EXTERNAL', 'BTC', 'ETH']:
            if entity_name in observation.states:
                ext_state = observation.states[entity_name]
                # Try to extract price from different possible attributes
                for attr in ['price', 'mark_price', 'close_price']:
                    if hasattr(ext_state, attr):
                        external_price = getattr(ext_state, attr)
                        break
                if external_price is not None:
                    break
                    
        if external_price is not None:
            self.external_prices.append(external_price)
            
        if len(self.uniswap_prices) >= 20 and len(self.external_prices) >= 20:
            self._calculate_correlation()

    def _calculate_correlation(self):
        """Calculate correlation between Uniswap price and external instrument"""
        # Convert to numpy arrays
        uni_prices = np.array(list(self.uniswap_prices))
        ext_prices = np.array(list(self.external_prices))
        
        min_len = min(len(uni_prices), len(ext_prices))
        uni_prices = uni_prices[-min_len:]
        ext_prices = ext_prices[-min_len:]
        
        uni_returns = np.diff(np.log(uni_prices))
        ext_returns = np.diff(np.log(ext_prices))
        
        if len(uni_returns) < 2:
            return
            
        correlation = np.corrcoef(uni_returns, ext_returns)[0, 1]
        
        if np.isnan(correlation):
            correlation = 0
            
        self.current_correlation = correlation
        self._debug(f"Current correlation: {correlation:.4f}")
        
        self._adjust_tau()

    def _adjust_tau(self):
        """Adjust tau based on current correlation value"""
        if self.current_correlation is None:
            return
            
        abs_corr = abs(self.current_correlation)
        
        if abs_corr >= self._params.CORRELATION_THRESHOLD:
            scale_factor = self._params.HIGH_CORR_SCALE
            self._debug(f"High correlation detected ({abs_corr:.4f}), narrowing range")
        else:
            scale_factor = self._params.LOW_CORR_SCALE
            self._debug(f"Low correlation detected ({abs_corr:.4f}), widening range")
            
        self.adjusted_tau = self._params.BASE_TAU * scale_factor
        
        self.adjusted_tau = max(self._params.MIN_TAU, 
                               min(self._params.MAX_TAU, self.adjusted_tau))
                               
        self._debug(f"Adjusted tau: {self._params.BASE_TAU} -> {self.adjusted_tau}")

    def predict(self) -> List[ActionToTake]:
        """Main strategy logic"""
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        current_price = uniswap_entity.global_state.price
        
        self.uniswap_prices.append(current_price)
        
        if len(self.uniswap_prices) >= 20 and len(self.external_prices) >= 20:
            self._calculate_correlation()
        
        if not uniswap_entity.is_position and not self.deposited_initial_funds:
            self._debug("No active position. Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()
            
        if not uniswap_entity.is_position:
            self._debug("No active position. Running first rebalance")
            return self._rebalance()
            
        lower_bound = uniswap_entity.internal_state.price_lower
        upper_bound = uniswap_entity.internal_state.price_upper
        
        if current_price < lower_bound or current_price > upper_bound:
            self._debug(f"Price {current_price} moved outside range [{lower_bound}, {upper_bound}]")
            return self._rebalance()
            
        return []

    def _deposit_to_lp(self) -> List[ActionToTake]:
        """Deposit initial funds"""
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]

    def _rebalance(self) -> List[ActionToTake]:
        """Rebalance positions with correlation-adjusted tau"""
        actions = []
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        
        if entity.is_position and entity.internal_state.liquidity > 0:
            actions.append(
                ActionToTake(entity_name='UNISWAP_V3', action=Action(action='close_position', args={}))
            )
            self._debug("Closed existing position")
            
        tau = self.adjusted_tau  # Use correlation-adjusted tau
        current_price = entity.global_state.price
        tick_spacing = self.tick_spacing
        
        price_lower = current_price * 1.0001 ** (-tau * tick_spacing)
        price_upper = current_price * 1.0001 ** (tau * tick_spacing)
        
        delegate_get_cash = lambda obj: obj.get_entity('UNISWAP_V3').internal_state.cash
        actions.append(ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(
                action='open_position',
                args={
                    'amount_in_notional': delegate_get_cash,  # All available cash
                    'price_lower': price_lower,
                    'price_upper': price_upper
                }
            )
        ))
        
        correlation_info = ""
        if self.current_correlation is not None:
            correlation_info = f" (correlation: {self.current_correlation:.2f})"
            
        return actions