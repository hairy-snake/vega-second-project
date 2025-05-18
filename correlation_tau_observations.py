import os
from typing import List, Dict, Optional
from datetime import datetime, UTC

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3 import (
    UniswapV3EthereumPoolHourDataLoader, EthereumUniswapV3Loader, UniswapV3EthereumPoolMinuteDataLoader
)
from fractal.loaders.binance import BinanceHourPriceLoader, BinanceMinutePriceLoader
from fractal.loaders.structs import PriceHistory, PoolHistory

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState, GlobalState
from correlation_tau_strategy import CorrelationTauStrategy, CorrelationTauParams

# Load environment variables
load_dotenv()


class ExternalPriceState(GlobalState):
    """Global state for external price reference"""
    def __init__(self, price: float):
        self.price = price


def get_correlation_observations(
        pool_data: PoolHistory, 
        uniswap_prices: PriceHistory, 
        external_prices: PriceHistory,
        external_instrument: str = "EXTERNAL",
        start_time: datetime = None, 
        end_time: datetime = None
    ) -> List[Observation]:
    """
    Get observations for the CorrelationTauStrategy with both Uniswap and external price data.

    Args:
        pool_data: Pool data history from Uniswap
        uniswap_prices: Price history for Uniswap pool
        external_prices: Price history for external reference instrument
        external_instrument: Name of the external instrument
        start_time: Start time for observations
        end_time: End time for observations

    Returns:
        List of Observation objects with both Uniswap and external price data
    """
    observations_df = pool_data.join(uniswap_prices)
    observations_df = observations_df.dropna()
    
    if external_prices is not None:
        external_df = external_prices.copy()
        external_df.columns = ['external_price']
        observations_df = observations_df.join(external_df)
    
    if start_time is not None or end_time is not None:
        observations_df = observations_df.loc[start_time:end_time]
    
    observations_df = observations_df[observations_df.tvl > 0]
    observations_df = observations_df.sort_index()
    
    result = []
    for timestamp, row in observations_df.iterrows():
        states = {
            'UNISWAP_V3': UniswapV3LPGlobalState(
                price=row.price, 
                tvl=row.tvl, 
                volume=row.volume, 
                fees=row.fees, 
                liquidity=row.liquidity
            )
        }
        
        if 'external_price' in row and not pd.isna(row.external_price):
            states[external_instrument] = ExternalPriceState(price=row.external_price)
            
        result.append(Observation(timestamp=timestamp, states=states))
        
    return result


def build_correlation_observations(
        ticker: str, 
        pool_address: str, 
        api_key: str,
        external_ticker: str = None,
        external_instrument: str = "EXTERNAL",
        start_time: datetime = None, 
        end_time: datetime = None, 
        fidelity: str = 'hour'
    ) -> List[Observation]:
    """
    Build observations for the CorrelationTauStrategy with external price reference.

    Args:
        ticker: Trading pair for Uniswap pool
        pool_address: Uniswap V3 pool address
        api_key: The Graph API key
        external_ticker: Trading pair for external reference (if different from pool ticker)
        external_instrument: Name to use for external instrument in observation
        start_time: Start time for data
        end_time: End time for data
        fidelity: Data resolution ('hour' or 'minute')

    Returns:
        List of Observation objects
    """
    if external_ticker is None:
        external_ticker = ticker
        
    if fidelity == 'hour':
        pool_data: PoolHistory = UniswapV3EthereumPoolHourDataLoader(
            api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        uniswap_prices: PriceHistory = BinanceHourPriceLoader(
            ticker, loader_type=LoaderType.CSV).read(with_run=True)
        external_prices: PriceHistory = BinanceHourPriceLoader(
            external_ticker, loader_type=LoaderType.CSV).read(with_run=True)
    elif fidelity == 'minute':
        pool_data: PoolHistory = UniswapV3EthereumPoolMinuteDataLoader(
            api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        uniswap_prices: PriceHistory = BinanceMinutePriceLoader(
            ticker, loader_type=LoaderType.CSV, start_time=start_time, end_time=end_time).read(with_run=True)
        external_prices: PriceHistory = BinanceMinutePriceLoader(
            external_ticker, loader_type=LoaderType.CSV, start_time=start_time, end_time=end_time).read(with_run=True)
    else:
        raise ValueError("Fidelity must be either 'hour' or 'minute'.")
        
    return get_correlation_observations(
        pool_data, uniswap_prices, external_prices, 
        external_instrument, start_time, end_time
    )


if __name__ == '__main__':
    ticker = "ETHUSDT"
    pool_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    external_ticker = "BTCUSDT"  # Use BTC as external reference
    external_instrument = "BTC"
    
    THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')
    if not THE_GRAPH_API_KEY:
        raise ValueError("THE_GRAPH_API_KEY environment variable not set")
        
    token0_decimals, token1_decimals = EthereumUniswapV3Loader(
        THE_GRAPH_API_KEY, loader_type=LoaderType.CSV).get_pool_decimals(pool_address)
    
    params = CorrelationTauParams(
        BASE_TAU=10.0,
        INITIAL_BALANCE=1_000_000,
        CORRELATION_WINDOW=168,  # 1 week of hourly data
        MIN_TAU=5.0,
        MAX_TAU=20.0,
        CORRELATION_THRESHOLD=0.7,
        HIGH_CORR_SCALE=0.7,  # Reduce tau for high correlation
        LOW_CORR_SCALE=1.3    # Increase tau for low correlation
    )
    
    CorrelationTauStrategy.token0_decimals = token0_decimals
    CorrelationTauStrategy.token1_decimals = token1_decimals
    CorrelationTauStrategy.tick_spacing = 60
    strategy = CorrelationTauStrategy(debug=True, params=params)
        
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2024, 3, 1, tzinfo=UTC)
    
    observations = build_correlation_observations(
        ticker=ticker,
        pool_address=pool_address,
        api_key=THE_GRAPH_API_KEY,
        external_ticker=external_ticker,
        external_instrument=external_instrument,
        start_time=start_time,
        end_time=end_time,
        fidelity='hour'
    )
    
    print(f"Loaded {len(observations)} observations")
    

    result = strategy.run(observations)
    
    print(result.get_default_metrics())
    
    result.to_dataframe().to_csv('correlation_tau_strategy_result.csv')
    print(result.to_dataframe().iloc[-1])
