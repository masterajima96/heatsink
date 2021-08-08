

import numpy as np

from hotrl.envs.house import House
from hotrl.envs.wrappers import FullyObsWrapper

size =4
inside_temp = 15.
outside_temp = 5.
temperatures = np.pad(
    np.full((size - 2, size - 2), fill_value=inside_temp),
    pad_width=[(1, 1), (1, 1)],
    mode='constant',
    constant_values=outside_temp
)
env = FullyObsWrapper(House(
    size=size, homies_params=[{'initial_room': 'Bedroom'}],
    temperatures=temperatures
))

n_episodes = 10000
home_info = {'home': {'room': 'Bedroom'}}
action = 0
for n in range(n_episodes):
    obs, reward, done, home_info = env.step(action)
    info = home_info[env.homies[0]]
    if info["temperature"] < info["comfort"][1]:
        print(f"Heating home in {info['room']}. "
              f"Temperature is {info['temperature']}. "
              f"Time is {info['dt']}")
        if info['room'] == 'Outside':
            env.render(temperature=True)
            continue
    action = env.action_names.index(f"heat_{info['room']}")
    env.render(temperature=True)
