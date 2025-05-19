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
from atr_tau_strategy import ATRTauResetParams, ATRTauResetStrategy
from tau_observations import build_observations, get_observations

load_dotenv()

THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')

if __name__ == '__main__':
    # Set up
    ticker: str = 'ETHUSDT'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')

    # Load data
    token0_decimals, token1_decimals = EthereumUniswapV3Loader(
        THE_GRAPH_API_KEY, loader_type=LoaderType.CSV).get_pool_decimals(pool_address)

    # Init the strategy
    params: ATRTauResetParams = ATRTauResetParams(TAU=10, INITIAL_BALANCE=1_000_000, TAU_SCALING_FACTOR=1.8, MIN_TAU=5.0, MAX_TAU=20.0, ATR_PERIOD=24)
    ATRTauResetStrategy.token0_decimals = token0_decimals
    ATRTauResetStrategy.token1_decimals = token1_decimals
    ATRTauResetStrategy.tick_spacing = 60
    strategy: ATRTauResetStrategy = ATRTauResetStrategy(debug=True, params=params)

    # Build observations
    entities = strategy.get_all_available_entities().keys()
    observations: List[Observation] = build_observations(
        ticker=ticker, pool_address=pool_address, api_key=THE_GRAPH_API_KEY,
        start_time=datetime(2024, 1, 11, tzinfo=UTC), end_time=datetime(2025, 1, 11, tzinfo=UTC),
        fidelity='hour'
    )
    observation0 = observations[0]
    # check if the observation has the right entities
    assert all(entity in observation0.states for entity in entities)

    # Run the strategy
    result = strategy.run(observations)

    print(result.get_default_metrics())  # show metrics
    result.to_dataframe().to_csv('atr_tau_strategy_result.csv')  # save results of strategy states
    print(result.to_dataframe().iloc[-1])  # show the last state of the strategy