import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from datetime import datetime, UTC
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from correlation_tau_strategy import CorrelationTauStrategy, CorrelationTauParams
from correlation_tau_observations import build_correlation_observations


def build_grid():
    """Build parameter grid for correlation tau strategy"""
    return ParameterGrid({
        'BASE_TAU': np.arange(start=5, stop=15, step=1, dtype=int),
        'MIN_TAU': [5.0],
        'MAX_TAU': [25.0],
        'CORRELATION_THRESHOLD': [0.6, 0.7, 0.8],
        'HIGH_CORR_SCALE': [0.6, 0.7, 0.8],  # Narrower range when correlation is high
        'LOW_CORR_SCALE': [1.2, 1.4],        # Wider range when correlation is low
        'CORRELATION_WINDOW': [168],         # 1 week of hourly data
        'INITIAL_BALANCE': [1_000_000]
    })


if __name__ == '__main__':
    ticker = "ETHUSDT"
    pool_address = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"
    external_ticker = "BTCUSDT"  # Use BTC as correlation reference
    
    start_time = datetime(2024, 1, 1, tzinfo=UTC)
    end_time = datetime(2025, 1, 1, tzinfo=UTC)
    
    fidelity = 'hour'
    experiment_name = f'correlation_tau_{fidelity}_{ticker}_with_{external_ticker.split("USDT")[0]}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'
    
    CorrelationTauStrategy.token0_decimals = 6  # USDT
    CorrelationTauStrategy.token1_decimals = 18  # ETH
    CorrelationTauStrategy.tick_spacing = 60
    
    THE_GRAPH_API_KEY = os.getenv('THE_GRAPH_API_KEY')
    if not THE_GRAPH_API_KEY:
        raise ValueError("THE_GRAPH_API_KEY environment variable not set")
    
    mlflow_config = MLFlowConfig(
        mlflow_uri=os.getenv('MLFLOW_URI'),
        experiment_name=experiment_name,
    )
    
    observations = build_correlation_observations(
        ticker=ticker,
        pool_address=pool_address,
        api_key=THE_GRAPH_API_KEY,
        external_ticker=external_ticker,
        external_instrument="BTC",  # Will be used in the strategy
        start_time=start_time,
        end_time=end_time,
        fidelity=fidelity
    )
    

    experiment_config = ExperimentConfig(
        strategy_type=CorrelationTauStrategy,
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
    print(f"Experiment complete: {experiment_name}")
    print(f"View results at: {mlflow_config.mlflow_uri}/#/experiments") 