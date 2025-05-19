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
    price_lower: float
    price_upper: float
    allocation: float
    position_id: Optional[int] = None


@dataclass
class DistributionTauParams(BaseStrategyParams):
    BASE_TAU: float
    INITIAL_BALANCE: float
    NUM_POSITIONS: int = 3
    HISTORY_LENGTH: int = 500
    USE_ATR: bool = True
    ATR_PERIOD: int = 14
    MIN_TAU: float = 5.0
    MAX_TAU: float = 20.0


class DistributionTauStrategy(BaseStrategy):
    token0_decimals: int = -1
    token1_decimals: int = -1
    tick_spacing: int = -1

    def __init__(self, params: DistributionTauParams, debug: bool = False, *args, **kwargs):
        self._params: DistributionTauParams = None
        assert self.token0_decimals != -1 and self.token1_decimals != -1 and self.tick_spacing != -1
        super().__init__(params=params, debug=debug, *args, **kwargs)
        self.deposited_initial_funds = False
        
        self.price_history = deque(maxlen=self._params.HISTORY_LENGTH)
        self.volume_history = deque(maxlen=self._params.HISTORY_LENGTH)
        self.atr = None
        self.active_positions = {}
        self.last_rebalance_price = None
        self.last_rebalance_time = None
        self.is_rebalancing = False
        self.min_rebalance_interval = 24
        self.price_movement_threshold = 0.05
        self._debug("Strategy initialized with params: " + str(params))

    def set_up(self):
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

    def _update_market_data(self, current_price: float, current_volume: float, timestamp=None):
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        if self._params.USE_ATR and len(self.price_history) > self._params.ATR_PERIOD:
            self._update_atr()

    def _update_atr(self):
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
        if len(self.price_history) < 30 or len(self.volume_history) < 30:
            return []
            
        prices = np.array(list(self.price_history))
        volumes = np.array(list(self.volume_history))
        
        min_price = np.min(prices)
        max_price = np.max(prices)
        
        num_bins = min(20, len(prices) // 5)
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

    def _get_adjusted_tau(self) -> float:
        base_tau = self._params.BASE_TAU
        
        if self._params.USE_ATR and self.atr is not None and self.price_history:
            current_price = self.price_history[-1]
            rel_atr = self.atr / current_price
            
            if rel_atr > 0.03:
                adjustment = 1.5 + 25 * rel_atr
                self._debug(f"High volatility detected ({rel_atr:.2%}), widening position ranges")
            else:
                adjustment = 1.0 + 15 * rel_atr
                
            adjusted_tau = base_tau * adjustment
            adjusted_tau = max(self._params.MIN_TAU, min(self._params.MAX_TAU, adjusted_tau))
            self._debug(f"Adjusted tau: {base_tau:.2f} -> {adjusted_tau:.2f} based on volatility ({rel_atr:.2%})")
            return adjusted_tau
        
        self._debug(f"Using base tau: {base_tau}")
        return base_tau

    def _create_positions(self, current_price: float) -> List[Position]:
        tau = self._get_adjusted_tau()
        num_positions = self._params.NUM_POSITIONS
        
        positions_info = []
        
        if num_positions == 1:
            positions_info.append({
                "price_lower": current_price * 1.0001 ** (-tau * self.tick_spacing / 2),
                "price_upper": current_price * 1.0001 ** (tau * self.tick_spacing / 2),
                "allocation": 1.0
            })
        else:
            position_weights = []
            std_dev = 0.8
            
            for i in range(num_positions):
                z_score = (i - (num_positions - 1) / 2) / std_dev
                weight = np.exp(-0.5 * z_score * z_score)
                position_weights.append(weight)
            
            total_weight = sum(position_weights)
            position_weights = [w / total_weight for w in position_weights]
            
            trend_direction = 0
            if len(self.price_history) > 30:
                recent_prices = list(self.price_history)[-30:]
                first_half_avg = sum(recent_prices[:15]) / 15
                second_half_avg = sum(recent_prices[15:]) / 15
                if second_half_avg > first_half_avg * 1.01:
                    trend_direction = 1
                    self._debug("Uptrend detected, skewing positions higher")
                elif second_half_avg < first_half_avg * 0.99:
                    trend_direction = -1
                    self._debug("Downtrend detected, skewing positions lower")
            
            for i in range(num_positions):
                position_width_factor = 0.8 + 0.4 * abs(i - (num_positions - 1) / 2) / ((num_positions - 1) / 2)
                position_tau = tau * position_width_factor
                
                center_adjustment = 0
                if trend_direction != 0:
                    center_pos = (num_positions - 1) / 2
                    distance_from_center = abs(i - center_pos)
                    max_distance = center_pos
                    skew_factor = 0.02 * (1 - distance_from_center / max_distance) * trend_direction
                    center_adjustment = current_price * skew_factor
                
                position_center = current_price + center_adjustment
                price_lower = position_center * 1.0001 ** (-position_tau * self.tick_spacing / 2)
                price_upper = position_center * 1.0001 ** (position_tau * self.tick_spacing / 2)
                
                positions_info.append({
                    "price_lower": price_lower, 
                    "price_upper": price_upper,
                    "allocation": position_weights[i]
                })
                
                self._debug(f"Calculated position {i+1}/{num_positions}: range [{price_lower:.2f}, {price_upper:.2f}] "
                           f"with {position_weights[i]:.2%} allocation")
        
        min_lower = min(pos["price_lower"] for pos in positions_info)
        max_upper = max(pos["price_upper"] for pos in positions_info)
        
        combined_position = Position(
            price_lower=min_lower,
            price_upper=max_upper,
            allocation=1.0
        )
        
        self._debug(f"Created combined position with range [{combined_position.price_lower:.2f}, {combined_position.price_upper:.2f}] "
                  f"representing {num_positions} conceptual positions")
        
        self.conceptual_positions = positions_info
        
        return [combined_position]

    def _create_default_positions(self, current_price: float) -> List[Position]:
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

    def _should_rebalance(self, current_price: float, current_time=None) -> bool:
        if not self.active_positions:
            self._debug("No active positions, should rebalance")
            return True
            
        if current_time is not None and self.last_rebalance_time is not None:
            hours_since_last_rebalance = (current_time - self.last_rebalance_time).total_seconds() / 3600
            if hours_since_last_rebalance < self.min_rebalance_interval:
                self._debug(f"Only {hours_since_last_rebalance:.1f} hours since last rebalance, minimum is {self.min_rebalance_interval}")
                return False
        
        for pos in self.active_positions.values():
            effective_lower = pos.price_lower * 1.10
            effective_upper = pos.price_upper * 0.90
            
            if effective_lower <= current_price <= effective_upper:
                return False
            else:
                if pos.price_lower <= current_price <= pos.price_upper:
                    if self.last_rebalance_price:
                        price_move = abs(current_price - self.last_rebalance_price) / self.last_rebalance_price
                        if price_move < self.price_movement_threshold:
                            self._debug(f"Price {current_price:.2f} outside effective range but price movement ({price_move:.2%}) below threshold")
                            return False
                
        self._debug(f"Price {current_price:.2f} outside position effective range, should rebalance")
        return True

    def predict(self) -> List[ActionToTake]:
        uniswap_entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        global_state = uniswap_entity.global_state
        current_price = global_state.price
        current_volume = global_state.volume
        current_time = getattr(self, '_current_observation_timestamp', None)
        
        self._update_market_data(current_price, current_volume, current_time)

        if not self.deposited_initial_funds:
            self._debug("Depositing initial funds...")
            self.deposited_initial_funds = True
            return self._deposit_to_lp()

        if self.is_rebalancing:
            if uniswap_entity.is_position:
                self._debug("Waiting for position to close...")
                return []
                
            self._debug("Opening new positions after rebalance...")
            actions = self._open_positions(current_price)
            if not actions:
                self._debug("No actions returned from _open_positions, ending rebalance")
                self.is_rebalancing = False
            return actions
            
        if self._should_rebalance(current_price, current_time):
            self._debug("Starting rebalance process...")
            self.is_rebalancing = True
            return self._close_positions()
            
        return []

    def _close_positions(self) -> List[ActionToTake]:
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')
        if entity.is_position:
            self._debug("Closing existing position")
            return [ActionToTake(
                entity_name='UNISWAP_V3',
                action=Action(action='close_position', args={})
            )]
        
        self._debug("No position to close, proceeding to open new positions")
        return self._open_positions(entity.global_state.price)

    def _open_positions(self, current_price: float) -> List[ActionToTake]:
        actions = []
        entity: UniswapV3LPEntity = self.get_entity('UNISWAP_V3')

        if entity.is_position:
            self._debug("Position still open, waiting...")
            return []

        positions = self._create_positions(current_price)
        if not positions:
            self._debug("No positions created")
            self.is_rebalancing = False
            return actions

        available_cash = entity.internal_state.cash
        self._debug(f"Available cash: {available_cash}")
        
        if available_cash <= 0:
            self._debug("No cash available for new positions")
            self.is_rebalancing = False
            return actions

        position = positions[0]
        position_cash = available_cash * position.allocation
        
        if position_cash > 0:
            self._debug(f"Opening combined position: range [{position.price_lower:.2f}, {position.price_upper:.2f}] "
                      f"with {position_cash:.2f} cash")
                      
            action = ActionToTake(
                entity_name='UNISWAP_V3',
                action=Action(
                    action='open_position',
                    args={
                        'amount_in_notional': position_cash,
                        'price_lower': position.price_lower,
                        'price_upper': position.price_upper
                    }
                )
            )
            actions.append(action)
            
            self.active_positions[1] = position
            self.last_rebalance_price = current_price
            self.last_rebalance_time = getattr(self, '_current_observation_timestamp', None)
        else:
            self._debug("Position cash is 0 or negative, skipping position opening")
            
        self.is_rebalancing = False
        return actions

    def _deposit_to_lp(self) -> List[ActionToTake]:
        return [ActionToTake(
            entity_name='UNISWAP_V3',
            action=Action(action='deposit', args={'amount_in_notional': self._params.INITIAL_BALANCE})
        )]

    def _rebalance(self, current_price: float) -> List[ActionToTake]:
        if not self.is_rebalancing:
            self.is_rebalancing = True
            return self._close_positions()
        else:
            return self._open_positions(current_price)

    def step(self, observation: Observation) -> List[ActionToTake]:
        self._current_observation_timestamp = observation.timestamp
        return super().step(observation)