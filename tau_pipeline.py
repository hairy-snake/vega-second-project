
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from datetime import datetime, UTC
from sklearn.model_selection import ParameterGrid

from fractal.core.pipeline import (
    DefaultPipeline, MLFlowConfig, ExperimentConfig)

from tau_strategy import TauResetStrategy
from tau_observations import build_observations


def build_grid():
    return ParameterGrid({
        'TAU': np.linspace(start=1, stop=15, num=15, dtype=int),
        'INITIAL_BALANCE': [1_000_000]
    })


if __name__ == '__main__':
    THE_GRAPH_API_KEY = os.getenv("THE_GRAPH_API_KEY")
    ticker: str = 'ETHUSDT'
    pool_address: str = '0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8'
    start_time = datetime(2024, 7, 1, tzinfo=UTC)
    end_time = datetime(2024, 9, 30, tzinfo=UTC)
    fidelity = 'hour'
    experiment_name = f'rtau_{fidelity}_{ticker}_{pool_address}_{start_time.strftime("%Y-%m-%d")}_{end_time.strftime("%Y-%m-%d")}'
    TauResetStrategy.token0_decimals = 6
    TauResetStrategy.token1_decimals = 18
    TauResetStrategy.tick_spacing = 60

    # Define MLFlow and Experiment configurations
    mlflow_config: MLFlowConfig = MLFlowConfig(
        mlflow_uri=os.getenv('MLFLOW_URI'),
        experiment_name=experiment_name,
    )
    observations = build_observations(ticker, pool_address, THE_GRAPH_API_KEY, start_time, end_time, fidelity=fidelity)
    assert len(observations) > 0
    experiment_config: ExperimentConfig = ExperimentConfig(
        strategy_type=TauResetStrategy,
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