import os
from typing import List
from datetime import datetime, UTC

import pandas as pd
from dotenv import load_dotenv

from fractal.loaders.base_loader import LoaderType
from fractal.loaders.thegraph.uniswap_v3 import (
    UniswapV3EthereumPoolHourDataLoader, EthereumUniswapV3Loader, UniswapV3EthereumPoolMinuteDataLoader
)
from fractal.loaders.binance import BinanceHourPriceLoader, BinanceMinutePriceLoader
from fractal.loaders.structs import PriceHistory, PoolHistory

from fractal.core.base import Observation
from fractal.core.entities import UniswapV3LPGlobalState
from distribution_tau_strategy import DistributionTauStrategy, DistributionTauParams

# Load environment variables
load_dotenv()


def build_distribution_observations(
        ticker: str, 
        pool_address: str, 
        api_key: str,
        start_time: datetime = None, 
        end_time: datetime = None,
        fidelity: str = 'hour',
        min_volume: float = 0.0,
        min_tvl: float = 0.0
    ) -> List[Observation]:
    """
    Build observations for the DistributionTauStrategy.
    
    Args:
        ticker: Trading pair symbol (e.g., 'ETHUSDT')
        pool_address: Uniswap V3 pool address
        api_key: The Graph API key
        start_time: Start time for observations
        end_time: End time for observations
        fidelity: Data resolution ('hour' or 'minute')
        min_volume: Minimum volume to include observation
        min_tvl: Minimum TVL to include observation
        
    Returns:
        List of Observation objects
    """
    if fidelity == 'hour':
        pool_data: PoolHistory = UniswapV3EthereumPoolHourDataLoader(
            api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        prices: PriceHistory = BinanceHourPriceLoader(
            ticker, loader_type=LoaderType.CSV).read(with_run=True)
    elif fidelity == 'minute':
        pool_data: PoolHistory = UniswapV3EthereumPoolMinuteDataLoader(
            api_key, pool_address, loader_type=LoaderType.CSV).read(with_run=True)
        prices: PriceHistory = BinanceMinutePriceLoader(
            ticker, loader_type=LoaderType.CSV, start_time=start_time, end_time=end_time).read(with_run=True)
    else:
        raise ValueError("Fidelity must be either 'hour' or 'minute'.")
        
    observations_df = pool_data.join(prices)
    observations_df = observations_df.dropna()
    
    if start_time is not None or end_time is not None:
        observations_df = observations_df.loc[start_time:end_time]
        
    if min_volume > 0:
        observations_df = observations_df[observations_df.volume >= min_volume]
    if min_tvl > 0:
        observations_df = observations_df[observations_df.tvl >= min_tvl]
        
    observations_df = observations_df.sort_index()
    
    result = []
    for timestamp, row in observations_df.iterrows():
        result.append(
            Observation(
                timestamp=timestamp,
                states={
                    'UNISWAP_V3': UniswapV3LPGlobalState(
                        price=row.price,
                        tvl=row.tvl,
                        volume=row.volume,
                        fees=row.fees,
                        liquidity=row.liquidity
                    )
                }
            )
        )
        
    return result


if __name__ == '__main__':
    ticker = "ETHUSDT"
    pool_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    
    THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')
    if not THE_GRAPH_API_KEY:
        raise ValueError("THE_GRAPH_API_KEY environment variable not set")
        
    token0_decimals, token1_decimals = EthereumUniswapV3Loader(
        THE_GRAPH_API_KEY, loader_type=LoaderType.CSV).get_pool_decimals(pool_address)
        
    params = DistributionTauParams(
        BASE_TAU=4.0,  # Tighter base range for better fee concentration
        INITIAL_BALANCE=1_000_000,
        NUM_POSITIONS=3,  # Multiple conceptual positions for normal distribution
        USE_ATR=True,
        ATR_PERIOD=14,
        MIN_TAU=2.0,  # Allows tighter ranges in low volatility
        MAX_TAU=20.0,  # Allows wider ranges in high volatility
        HISTORY_LENGTH=50  # Enough history for trend analysis
    )
    
    DistributionTauStrategy.token0_decimals = token0_decimals
    DistributionTauStrategy.token1_decimals = token1_decimals
    DistributionTauStrategy.tick_spacing = 60
    strategy = DistributionTauStrategy(debug=True, params=params)
    
    # Adjusted time range for better data coverage
    start_time = datetime(2024, 1, 11, tzinfo=UTC)
    end_time = datetime(2025, 1, 11, tzinfo=UTC)  # Reduced to 1 month for faster testing
    
    observations = build_distribution_observations(
        ticker=ticker,
        pool_address=pool_address,
        api_key=THE_GRAPH_API_KEY,
        start_time=start_time,
        end_time=end_time,
        fidelity='hour',
        min_volume=100.0  # Reduced from 1000.0 to include more data points
    )
    
    if not observations:
        raise ValueError("No observations loaded")
        
    print(f"Loaded {len(observations)} observations")
    
    result = strategy.run(observations)

    print("\nStrategy Metrics:")
    print(result.get_default_metrics())
    
    # Save results
    result.to_dataframe().to_csv('distribution_tau_strategy_result.csv')
    print("\nFinal State:")
    print(result.to_dataframe().iloc[-1])