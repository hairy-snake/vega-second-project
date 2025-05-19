import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime, UTC
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from timed_tau_strategy import TimedTauResetStrategy, TimedTauResetParams
from timed_tau_observations import build_observations


def build_grid():
    """Build parameter grid for timed tau reset strategy"""
    raw_grid = {
        'TAU': np.arange(start=5, stop=15, step=1, dtype=int),
        'INITIAL_BALANCE': [1_000_000],
        'EMA_PERIOD': [10, 20, 30],
        'REBALANCE_DELTA': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        'EMERGENCY_REBALANCE_THRESHOLD': [1.3, 1.5, 1.7]
    }
    
    return ParameterGrid(raw_grid)


if __name__ == '__main__':
    ticker = "ETHUSDT"
    pool_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2024, 3, 1, tzinfo=UTC)
    
    fidelity = 'hour'
    experiment_name = f'timed_rtau_{fidelity}_{ticker}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'
    
    TimedTauResetStrategy.token0_decimals = 6  # USDT
    TimedTauResetStrategy.token1_decimals = 18  # ETH
    TimedTauResetStrategy.tick_spacing = 60
    
    THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')
    if not THE_GRAPH_API_KEY:
        raise ValueError("THE_GRAPH_API_KEY environment variable not set")
    
    mlflow_config = MLFlowConfig(
        mlflow_uri=os.getenv('MLFLOW_URI'),
        experiment_name=experiment_name,
    )
    
    observations = build_observations(
        ticker=ticker,
        pool_address=pool_address,
        api_key=THE_GRAPH_API_KEY,
        start_time=start_time,
        end_time=end_time,
        fidelity=fidelity
    )
    
    if not observations:
        raise ValueError("No observations loaded")
    
    experiment_config = ExperimentConfig(
        strategy_type=TimedTauResetStrategy,
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
