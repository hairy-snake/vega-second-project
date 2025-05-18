from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats
from collections import deque

from fractal.core.base import (Action, ActionToTake, BaseStrategy,
                               BaseStrategyParams, NamedEntity, Observation)
from fractal.core.entities import UniswapV3LPConfig, UniswapV3LPEntity


@dataclass
class Position:
    """Represents a liquidity position in Uniswap V3"""
    price_lower: float
    price_upper: float
    allocation: float  # Percentage of total capital to allocate
    position_id: Optional[int] = None  # Assigned when position is opened


@dataclass
class DistributionTauParams(BaseStrategyParams):
    """
    Parameters for the distribution-based τ-reset strategy:
    - BASE_TAU: The base width of the price range
    - INITIAL_BALANCE: The initial balance for liquidity allocation
    - NUM_POSITIONS: Number of positions to maintain
    - HISTORY_LENGTH: Number of data points to keep for analysis
    - USE_ATR: Whether to adjust position width based on volatility
    - ATR_PERIOD: Period for ATR calculation
    - MIN_TAU: Minimum tau value
    - MAX_TAU: Maximum tau value
    """
    BASE_TAU: float
    INITIAL_BALANCE: float
    NUM_POSITIONS: int = 3
    HISTORY_LENGTH: int = 500
    USE_ATR: bool = True
    ATR_PERIOD: int = 14
    MIN_TAU: float = 5.0
    MAX_TAU: float = 20.0


class DistributionTauStrategy(BaseStrategy):
    """
    A τ-reset strategy that maintains multiple positions based on volume distribution.
    
    This strategy:
    1. Analyzes trading volume distribution across price ranges
    2. Places multiple positions with different widths and allocations
    3. Adjusts position widths based on market volatility using ATR
    """

    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(self, params: DistributionTauParams, debug: bool = False, *args, **kwargs):
        self._params: DistributionTauParams = None  # set for type hinting
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False
        
        # Initialize data structures
        self.price_history = deque(maxlen=self._params.HISTORY_LENGTH)
        self.volume_history = deque(maxlen=self._params.HISTORY_LENGTH)
        self.atr = None
        self.active_positions = {}
        self.last_rebalance_price = None

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

    def _update_market_data(self, current_price: float, current_volume: float):
        """Update price and volume history"""
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        if self._params.USE_ATR and len(self.price_history) > self._params.ATR_PERIOD:
            self._update_atr()

    def _update_atr(self):
        """Calculate Average True Range based on price history"""
        prices = list(self.price_history)
        
        true_ranges = []
        for i in range(1, min(len(prices), self._params.ATR_PERIOD + 1)):
            high = max(prices[-i], prices[-i-1])
            low = min(prices[-i], prices[-i-1])
            true_ranges.append(high - low)
            
        if not true_ranges:
            return
            
        new_atr = sum(true_ranges) / len(true_ranges)
        
        if self.atr is None:
            self.atr = new_atr
        else:
            self.atr = 0.85 * self.atr + 0.15 * new_atr
            
        current_price = prices[-1] if prices else 1.0
        self._debug(f"Updated ATR: {self.atr:.4f} ({self.atr/current_price:.2%} of price)")

    def _analyze_volume_distribution(self) -> List[Tuple[float, float, float]]:
        """
        Analyze volume distribution to determine optimal position ranges.
        Returns list of (price_lower, price_upper, allocation) tuples.
        """
        if len(self.price_history) < 30 or len(self.volume_history) < 30:
            return []
            
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        num_bins = min(20, len(prices) // 5)  # At most 20 bins
        bins = np.linspace(min_price, max_price, num_bins + 1)
        
        volume_per_bin = np.zeros(num_bins)
        for i in range(len(prices)):
            bin_idx = np.digitize(prices[i], bins) - 1
            if 0 <= bin_idx < num_bins:
                volume_per_bin[bin_idx] += volumes[i]
                
        total_volume = np.sum(volume_per_bin)
        if total_volume > 0:
            volume_pct = volume_per_bin / total_volume
        else:
            volume_pct = np.ones(num_bins) / num_bins
            
        sorted_indices = np.argsort(volume_pct)[::-1]
        
        num_positions = min(self._params.NUM_POSITIONS, len(sorted_indices))
        position_ranges = []
        
        total_selected_volume = sum(volume_pct[sorted_indices[:num_positions]])
        
        for i in range(num_positions):
            bin_idx = sorted_indices[i]
            price_lower = bins[bin_idx]
            price_upper = bins[bin_idx + 1]
            
            allocation = volume_pct[bin_idx] / total_selected_volume
            
            position_ranges.append((price_lower, price_upper, allocation))
            self._debug(f"Volume hotspot: [{price_lower:.2f}, {price_upper:.2f}] "
                       f"with {volume_pct[bin_idx]:.2%} of volume")
                       
        return position_ranges

    def _create_positions(self, current_price: float) -> List[Position]:
        """Create position configurations based on volume analysis and volatility"""
        volume_ranges = self._analyze_volume_distribution()
        
        if not volume_ranges:
            return self._create_default_positions(current_price)
            
        tau = self._get_adjusted_tau()
        
        positions = []
        used_allocation = 0.0
        
        for price_lower, price_upper, allocation in volume_ranges:
            mid_price = (price_lower + price_upper) / 2
            
            bin_width_ratio = (price_upper - price_lower) / mid_price
            position_tau = min(tau, bin_width_ratio * 10000 / self.tick_spacing)
            
            pos_lower = mid_price * 1.0001 ** (-position_tau * self.tick_spacing / 2)
            pos_upper = mid_price * 1.0001 ** (position_tau * self.tick_spacing / 2)
            
            positions.append(Position(
                price_lower=pos_lower,
                price_upper=pos_upper,
                allocation=allocation
            ))
            used_allocation += allocation
            
        has_current_price = any(p.price_lower <= current_price <= p.price_upper for p in positions)
        
        if not has_current_price:
            self._debug(f"Adding position to cover current price {current_price}")
            
            current_price_position = Position(
                price_lower=current_price * 1.0001 ** (-tau * self.tick_spacing / 2),
                price_upper=current_price * 1.0001 ** (tau * self.tick_spacing / 2),
                allocation=0.2  # 20% allocation
            )
            
            remaining_allocation = 1.0 - current_price_position.allocation
            for pos in positions:
                pos.allocation *= remaining_allocation / used_allocation
                
            positions.append(current_price_position)
            
        return positions

    def _create_default_positions(self, current_price: float) -> List[Position]:
        """Create default positions centered around current price when no volume data available"""
        num_positions = self._params.NUM_POSITIONS
        tau = self._get_adjusted_tau()
        
        positions = []
        allocation_per_position = 1.0 / num_positions
        
        for i in range(num_positions):
            position_tau = tau * (0.75 + 0.5 * i / (num_positions - 1 or 1))
            
            price_lower = current_price * 1.0001 ** (-position_tau * self.tick_spacing / 2)
            price_upper = current_price * 1.0001 ** (position_tau * self.tick_spacing / 2)
            
            positions.append(Position(
                price_lower=price_lower,
                price_upper=price_upper,
                allocation=allocation_per_position
            ))
            
        return positions

    def _get_adjusted_tau(self) -> float:
        """Calculate tau value adjusted for market conditions"""
        base_tau = self._params.BASE_TAU
        
        if self._params.USE_ATR and self.atr is not None and self.price_history:
            current_price = self.price_history[-1]
            
            rel_atr = self.atr / current_price
            
            adjustment = 1.0 + 20 * rel_atr
            adjusted_tau = base_tau * adjustment
            
            adjusted_tau = max(self._params.MIN_TAU, min(self._params.MAX_TAU, adjusted_tau))
            
            self._debug(f"Adjusted tau: {base_tau} -> {adjusted_tau} based on volatility ({rel_atr:.2%})")
            return adjusted_tau
            
        return base_tau

    def _should_rebalance(self, current_price: float) -> bool:
        """Determine if positions should be rebalanced"""
        if not self.active_positions:
            return True
            
        price_in_range = False
        for pos in self.active_positions.values():
            if pos.price_lower <= current_price <= pos.price_upper:
                price_in_range = True
                break
                
        if not price_in_range:
            self._debug(f"Rebalance triggered: price {current_price} outside all position ranges")
            return True
            
        if self.last_rebalance_price:
            price_move = abs(current_price - self.last_rebalance_price) / self.last_rebalance_price
            if price_move > 0.05:  # 5% price movement
                self._debug(f"Rebalance triggered: {price_move:.2%} price move since last rebalance")
                return True
                
        return False

    def predict(self) -> List[ActionToTake]:
        """Main strategy logic"""
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity.global_state
        current_price = global_state.price
        current_volume = global_state.volume
        
        self._update_market_data(current_price, current_volume)
        
        if not self.deposited_initial_funds:
            self._debug("Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()
            
        if self._should_rebalance(current_price):
            self._debug("Rebalancing positions...")
            return self._rebalance(current_price)
            
        return []

    def _deposit_to_lp(self) -> List[ActionToTake]:
        """Deposit initial funds"""
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]

    def _rebalance(self, current_price: float) -> List[ActionToTake]:
        """Rebalance positions based on current market conditions"""
        actions = []
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        
        if entity.is_position:
            self._debug("Closing existing position")
            actions.append(
                ActionToTake(entity_name='UNISWAP_V3', action=Action(action='close_position', args={}))
            )
        
        self.active_positions = {}
        
        new_positions = self._create_positions(current_price)
        
        total_cash = entity.internal_state.cash
        self._debug(f"Available cash: {total_cash:.2f}")
        
        next_position_id = 1
        for position in new_positions:
            position_amount = total_cash * position.allocation
            
            actions.append(
                ActionToTake(
                    entity_name='UNISWAP_V3',
                    action=Action(
                        action='open_position',
                        args={
                            'amount_in_notional': position_amount,
                            'price_lower': position.price_lower,
                            'price_upper': position.price_upper
                        }
                    )
                )
            )
            
            position.position_id = next_position_id
            self.active_positions[next_position_id] = position
            next_position_id += 1
            
            self._debug(f"Opened position with range [{position.price_lower:.4f}, {position.price_upper:.4f}] "
                       f"and {position.allocation:.2%} allocation ({position_amount:.2f})")
                       
        self.last_rebalance_price = current_price
        
        return actions