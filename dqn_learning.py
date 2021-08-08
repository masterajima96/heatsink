
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.trainer_template import build_trainer
from ray.tune.registry import register_env

from hotrl import EXPERIMENTS_DIR
from hotrl.envs.house import House
from hotrl.envs.house_logger import HouseLogger
from hotrl.envs.wrappers import FullyObsWrapper
from hotrl.rllib_experiments.trainables import MaintenancePolicy

size = 4
inside_temp = 15.
outside_temp = 5.

env_config = dict(
    size=size,
    homies_params=[{'initial_room': 'Bedroom'}],
    temperatures=np.pad(
        np.full((size - 2, size - 2), fill_value=inside_temp),
        pad_width=[(1, 1), (1, 1)],
        mode='constant',
        constant_values=outside_temp
    ),
    heat_model_config=dict(
        RSI=4.2 * 2,
        heater_output=1000,
    ),
    homie_reward_scaler=tune.function(lambda x: x ** 5 if x < 1 else x),
)

register_env("House", lambda config: FullyObsWrapper(House(**config)))
ray.init(
    local_mode=True,
)

trials = tune.run(
    # run_or_experiment=dqn_train,
    run_or_experiment=build_trainer(
        name="MaintenanceTrainer",
        default_policy=MaintenancePolicy),
    loggers=[HouseLogger],
    verbose=1,
    local_dir=EXPERIMENTS_DIR,
    config={
        "model"     : {
            # List of [out_channels, kernel, stride] for each filter
            "conv_filters": [
                [2, [4, 4], 1]
            ],
        },
        "env"       : "House",
        "env_config": env_config
    },
)