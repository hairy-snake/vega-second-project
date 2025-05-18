import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime, UTC
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from distribution_tau_strategy import DistributionTauStrategy, DistributionTauParams
from distribution_tau_observations import build_distribution_observations


def build_grid():
    """Build parameter grid for distribution tau strategy"""
    raw_grid = {
        'BASE_TAU': np.arange(start=8, stop=15, step=1, dtype=int),
        'MIN_TAU': [5.0],
        'MAX_TAU': [20.0],
        'NUM_POSITIONS': [2, 3, 4],
        'USE_ATR': [True],
        'ATR_PERIOD': [14],
        'HISTORY_LENGTH': [500],
        'INITIAL_BALANCE': [1_000_000]
    }
    
    return ParameterGrid(raw_grid)


if __name__ == '__main__':
    ticker = "ETHUSDT"
    pool_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2024, 3, 1, tzinfo=UTC)
    
    fidelity = 'hour'
    experiment_name = f'distribution_tau_{fidelity}_{ticker}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'
    
    DistributionTauStrategy.token0_decimals = 6  # USDT
    DistributionTauStrategy.token1_decimals = 18  # ETH
    DistributionTauStrategy.tick_spacing = 60
    
    THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')
    if not THE_GRAPH_API_KEY:
        raise ValueError("THE_GRAPH_API_KEY environment variable not set")
    
    mlflow_config = MLFlowConfig(
        mlflow_uri=os.getenv('MLFLOW_URI'),
        experiment_name=experiment_name,
    )
    
    observations = build_distribution_observations(
        ticker=ticker,
        pool_address=pool_address,
        api_key=THE_GRAPH_API_KEY,
        start_time=start_time,
        end_time=end_time,
        fidelity=fidelity,
        min_volume=1000.0  # Minimum volume to include
    )
    
    if not observations:
        raise ValueError("No observations loaded")
    
    experiment_config = ExperimentConfig(
        strategy_type=DistributionTauStrategy,
        backtest_observations=observations,
        window_size=24,  # 1 day rolling window for metrics
        params_grid=build_grid(),
        debug=True
    )
    
    pipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    
    pipeline.run()