
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from datetime import datetime, UTC
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from atr_tau_strategy import ATRTauResetStrategy
from tau_observations import build_observations


def build_grid():
    return ParameterGrid({
        'TAU': np.ndarray([10]),
        'MAX_TAU': np.linspace(start=15, stop=20, num=20, dtype=int),
        'MIN_TAU': np.linspace(start=1, stop=7, num=7, dtype=int),
        'ATR_PERIOD': np.linspace(start=10, stop=20, num=20, dtype=int),
        'INITIAL_BALANCE': [1_000_000]
    })


if __name__ == '__main__':
    THE_GRAPH_API_KEY = os.getenv("THE_GRAPH_API_KEY")
    ticker: str = 'ETHUSDT'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    start_time = datetime(2024, 7, 1, tzinfo=UTC)
    end_time = datetime(2024, 9, 30, tzinfo=UTC)
    fidelity = 'hour'
    experiment_name = f'atr_rtau_{fidelity}_{ticker}_{pool_address}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'
    ATRTauResetStrategy.token0_decimals = 6
    ATRTauResetStrategy.token1_decimals = 18
    ATRTauResetStrategy.tick_spacing = 60

    # Define MLFlow and Experiment configurations
    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri=os.getenv('MLFLOW_URI'),
        experiment_name=experiment_name,
    )
    observations = build_observations(ticker, pool_address, THE_GRAPH_API_KEY, start_time, end_time, fidelity=fidelity)
    assert len(observations) > 0
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=ATRTauResetStrategy,
        backtest_observations=observations,
        window_size=24,
        params_grid=build_grid(),
        debug=True,
    )
    pipeline: DefaultPipeline = DefaultPipeline(
        experiment_config=experiment_config,
        mlflow_config=mlflow_config
    )
    pipeline.run()