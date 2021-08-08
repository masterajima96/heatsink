import time
from datetime import datetime, timedelta
from typing import Dict, Callable, Tuple, List

import numpy as np
import termcolor
from gym import spaces

from gym_minigrid.envs.empty import MiniGridEnv, OBJECT_TO_IDX, COLOR_TO_IDX, \
    TEMPERATURES, COLORS, COLOR_NAMES
from gym_minigrid.minigrid import WorldObj, CELL_PIXELS, Grid, Floor, Wall, Door
from hotrl.heat_transfer.model import SimpleModel, RoomModel

xy_coord = Tuple[int, int]
RoomType = ['Kitchen', 'Bedroom', 'Bathroom', 'LivingRoom', 'Dungeon',
            'Outside']


class HeatingTile(Floor):

    def __init__(self, color='blue', temperature=20):
        super().__init__(color, temperature)


class Homie(WorldObj):

    def __init__(self, house: 'House', initial_room: RoomType = 'Bedroom'):
        super(Homie, self).__init__('ball', color='blue')
        self.house = house
        self.current_room = initial_room
        self.cur_pos = self._place_within_the_room()

    def _place_within_the_room(self) -> xy_coord:
        if self.current_room == 'Outside':
            return 0, 0
        else:
            if isinstance(self.house, House):
                coord = self.house.rooms[self.current_room][0]
            elif isinstance(self.house, MultiRoomHouse):
                room = self.house.rooms[
                    self.house.room_names.index(self.current_room)]
                coord = room.top[0] + 1, room.top[1] + 1
            return coord

    def get_preferred_temperature(self, timestamp: datetime = None) -> int:
        """ Query homie for a preferred temperature at a current location,
        at a given time of the day or season"""

        if self.current_room == 'Kitchen':
            temp = 20, 25
        elif self.current_room == 'Bathroom':
            temp = 22, 24
        elif self.current_room == 'Bedroom':
            temp = 18, 20
        elif self.current_room == 'LivingRoom':
            temp = 19, 24
        elif self.current_room == 'Outside':
            temp = -30, 30
        else:
            raise ValueError('Undefined room type')

        return temp

    def step(self, timestamp: datetime) -> RoomType:
        date = dict(
            year=timestamp.year,
            month=timestamp.month,
            day=timestamp.day
        )

        sleep = datetime(**dict(**date, hour=0))
        morning_bath = datetime(**dict(**date, hour=7))
        breakfast = datetime(**dict(**date, hour=7, minute=30))
        leave_for_work = datetime(**dict(**date, hour=8))
        dinner = datetime(**dict(**date, hour=18))
        study = datetime(**dict(**date, hour=19))
        evening_bath = datetime(**dict(**date, hour=23, minute=30))

        if sleep <= timestamp < morning_bath:
            self.current_room = 'Bedroom'
        elif morning_bath <= timestamp < breakfast:
            self.current_room = 'Bathroom'
        elif breakfast <= timestamp < leave_for_work:
            self.current_room = 'Kitchen'
        elif leave_for_work <= timestamp < dinner:
            self.current_room = 'Outside'
        elif dinner <= timestamp < study:
            self.current_room = 'Kitchen'
        elif study <= timestamp < evening_bath:
            self.current_room = 'LivingRoom'
        elif evening_bath <= timestamp:
            self.current_room = 'Bathroom'

        self.cur_pos = self._place_within_the_room()

    def render(self, r, temperature: bool = False):
        if not temperature:
            self._set_color(r, temperature)
            r.drawCircle(CELL_PIXELS * 0.5, CELL_PIXELS * 0.5, 10)
        else:
            c = TEMPERATURES[self.temperature] if temperature else COLORS[
                self.color]
            r.setColor(*c)
            r.drawPolygon([
                (1, CELL_PIXELS),
                (CELL_PIXELS, CELL_PIXELS),
                (CELL_PIXELS, 1),
                (1, 1)
            ])


class HouseGrid(Grid):
    """ A grid-world house with tenants and temperatures for each cell

    The third dimension of the grid array holds information about the
    object type, object color as well is object's current temperature.
    """

    def encode(self, vis_mask=None):

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='int8')
        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    if v is None:
                        assert ValueError
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                    else:
                        array[i, j, 0] = OBJECT_TO_IDX[v.type]
                        array[i, j, 1] = COLOR_TO_IDX[v.color]
                        array[i, j, 2] = v.temperature

        return array


class House(MiniGridEnv):

    def __init__(
            self,
            temperatures: np.ndarray,
            heat_model_config: Dict,
            size: int = 4,
            start_dt: datetime = datetime.now(),
            dt_delta: timedelta = timedelta(minutes=1),
            homies_params: List[Dict] = None,
            homie_reward_scaler: Callable = lambda x: x ** 2,
    ):
        """ A square grid environment upon which the thermostat can operate.

        Each cell defines a room.

        Actions are discrete integer values corresponding to the combinations
        of rooms that we want to start heating up.

        :param temperatures: initial temperature for each cell of the grid
        :param size: defines both width and height of the grid
        :param start_dt: timestamp to which the temperatures correspond to
        :param dt_delta: a timescale on which the can be controlled
        :param homies_params: parameters that are necessary to define homies
        :param homie_reward_scaler: a function that scales reward coming from
                                    comfort of the temperature
        """
        # Deal with objects within the house
        self.rooms = {
            'Kitchen': [(2, 2)],
            'Bathroom': [(2, 1)],
            'Bedroom': [(1, 2)],
            'LivingRoom': [(1, 1)],
            'Outside': [(0, i) for i in range(size)] +
                       [(i, 0) for i in range(size)] +
                       [(size - 1, i) for i in range(size)] +
                       [(i, size - 1) for i in range(size)]
        }
        self.homies = [Homie(self, **params) for params in homies_params]
        self.homie_reward_scaler = homie_reward_scaler

        # Prepare the weather data and the heat transfer model
        self.temperatures = temperatures
        self.weather_time_series = self._get_weather_time_series()
        print(self.weather_time_series)
        self.current_dt = start_dt - timedelta(
            seconds=start_dt.second,
            microseconds=start_dt.microsecond
        )
        self.timedelta = dt_delta
        self.model = SimpleModel(**heat_model_config)

        super().__init__(
            grid_size=size,
            max_steps=1000,
            see_through_walls=True,
        )

        # Override action space
        self.action_dict = {
            (0, 0): 'heat_Nothing',
        }
        for room in sorted(self.rooms):
            if room == 'Outside':
                continue
            self.action_dict[self.rooms[room][0]] = f'heat_{room}'
        self.action_space = spaces.Box(low=0, high=1, shape=(size, size))

    def _get_weather_time_series(self):
        """ A function that allows to use synthetic or realistic data in the
        heat transfer model """
        n = 1440
        signal = np.cos(np.pi * np.arange(n) / float(n / 2))
        return self.rescale_linear(signal, 20, 10)

    def reset(self):
        super().reset()
        for homie in self.homies:
            homie.step(self.current_dt)

    def _gen_grid(self, width, height):
        assert width == height

        # Create an empty grid
        self.grid = HouseGrid(width, height)

        # Generate walls
        self.grid.wall_rect(0, 0, width, height)

        # Place tenants in the house
        for homie in self.homies:
            self.grid.set(*homie.cur_pos, v=homie)

        self.place_agent()

        # Place the heating tiles in the house
        for i, cell in enumerate(self.grid.grid):
            x, y = divmod(i, width)
            if cell is None:
                self.grid.grid[i] = HeatingTile(
                    temperature=self.temperatures[x, y]
                )
            self.grid.grid[i].temperature = self.temperatures[x, y]

        self.mission = "it's getting hot in here"

    def reward(self, heating_cost: float, homie_discomfort: float):
        return -(homie_discomfort + heating_cost)

    def step(self, action):
        self.current_dt += self.timedelta
        self.step_count += 1

        # Reset the temperature outside of the house
        idx = self.current_dt.hour * 60 + self.current_dt.minute
        self.temperatures[tuple(zip(*self.rooms['Outside']))] = \
            self.weather_time_series[idx]

        # Move each homie and determine their preference for the temperature
        info = {
            'comfort_penalty': 0,
            'cost': 0,
            # 'extreme_penalty': 0,
        }
        for homie in self.homies:
            info[homie] = {}
            info[homie]["room"] = homie.current_room
            info[homie]["dt"] = self.current_dt
            info[homie]["temperature"] = self.temperatures[
                self.rooms[homie.current_room][0]]
            info[homie]["comfort"] = homie.get_preferred_temperature(
                self.current_dt)
            if not info[homie]["comfort"][0] <= \
                   info[homie]["temperature"] <= \
                   info[homie]["comfort"][1]:
                info['comfort_penalty'] += self.homie_reward_scaler(min(
                    abs(info[homie]["temperature"] - info[homie]["comfort"][0]),
                    abs(info[homie]["temperature"] - info[homie]["comfort"][1])
                ))
            homie.step(timestamp=self.current_dt)

        # Record the temperature for each room for visualization purposes
        for room, cells in self.rooms.items():
            info[f'{room}_temperature'] = self.temperatures[cells[0]]

        # Adjust the temperature in the house wrt to the preferences of homies
        self._change_temperature(action)

        # Calculate cost of heating by summing over cells
        info['cost'] = np.sum(action)

        # Calculate the reward
        reward = self.reward(info['comfort_penalty'], info['cost'])

        done = self.step_count >= self.max_steps
        # if self.temperatures.max() > 40:
        #     reward -= 100
        #     info['extreme_penalty'] = 100

        obs = self.gen_obs()

        # Remove the agent from the observation
        obs['image'][:, :, 0][obs['image'][:, :, 0] == 10] = 1

        # Logging
        if self.step_count % 100 == 0:
            colors = {
                'Kitchen': 'yellow',
                'Bathroom': 'magenta',
                'Bedroom': 'green',
                'LivingRoom': 'blue',
                'Outside': 'grey'
            }
            x = info[homie]
            info_dict = info.copy()
            del info_dict[homie]
            action_idx = next(zip(*np.where(action == action.max())))
            print(termcolor.colored(
                text={**info, **x, 'action': self.action_dict[action_idx]},
                color=colors[x['room']]
            ))
            print(self.temperatures)

        return obs, reward, done, info

    def _change_temperature(self, heatmap: np.ndarray = None):
        """ Changes the temperature of each object in the house """
        if heatmap is None:
            heatmap = np.zeros(self.temperatures.shape, dtype=float)
        self.temperatures = self.model.step(
            temperatures=self.temperatures,
            heat=heatmap
        )
        for i, cell in enumerate(self.grid.grid):
            x, y = divmod(i, self.grid.width)
            self.grid.grid[i].temperature = self.temperatures[x, y]

    @staticmethod
    def rescale_linear(array, new_min, new_max):
        """Rescale an arrary linearly."""
        minimum, maximum = np.min(array), np.max(array)
        m = (new_max - new_min) / (maximum - minimum)
        b = new_min - m * minimum
        return m * array + b


class Room:
    def __init__(self,
                 top,
                 size,
                 entryDoorPos,
                 exitDoorPos,
                 name,
                 temperature,
                 ):
        self.top = top
        self.size = size
        self.x1 = top[0]
        self.y1 = top[1]
        self.x2 = self.x1 + self.size[0]
        self.y2 = self.y1 + self.size[1]
        self.entryDoorPos = entryDoorPos
        self.exitDoorPos = exitDoorPos
        self.name = name
        self.temperature = temperature


class MultiRoomHouse(MiniGridEnv):
    """
    Environment with multiple rooms (subgoals)
    """

    def __init__(self,
                 t_out: float = -20,
                 t_start: float = 20,
                 start_dt: datetime = datetime.now(),
                 dt_delta: timedelta = timedelta(minutes=1),
                 homies_params: List[Dict] = None,
                 homie_reward_scaler: float = 1,
                 room_names: List[str] = RoomType,
                 minNumRooms=5,
                 maxNumRooms=5,
                 maxRoomSize=10,
                 seed=1337,
                 ):

        self.t_out = t_out
        self.t_start = t_start
        self.current_dt = start_dt
        self.timedelta = dt_delta
        self.homie_reward_scaler = homie_reward_scaler
        self.model = RoomModel()

        assert minNumRooms > 0
        assert maxNumRooms >= minNumRooms
        assert maxRoomSize >= 4

        self.room_names = room_names.copy()
        self.room_names.remove("Outside")
        self.minNumRooms = minNumRooms
        self.maxNumRooms = maxNumRooms
        self.maxRoomSize = maxRoomSize

        self.rooms = []

        super().__init__(
            grid_size=25,
            max_steps=self.maxNumRooms * 20,
            seed=seed
        )
        self.homies = [Homie(self, **params) for params in homies_params]
        for homie in self.homies:
            self.grid.set(*homie.cur_pos, v=homie)

    def _gen_grid(self, width, height):
        roomList = []

        # Choose a random number of rooms to generate
        numRooms = self._rand_int(self.minNumRooms, self.maxNumRooms + 1)

        while len(roomList) < numRooms:
            curRoomList = []

            entryDoorPos = (
                self._rand_int(0, width - 2),
                self._rand_int(0, width - 2)
            )

            # Recursively place the rooms
            self._placeRoom(
                numRooms,
                roomList=curRoomList,
                minSz=4,
                maxSz=self.maxRoomSize,
                entryDoorWall=2,
                entryDoorPos=entryDoorPos
            )

            if len(curRoomList) > len(roomList):
                roomList = curRoomList

        # Store the list of rooms in this environment
        assert len(roomList) > 0
        self.rooms = roomList

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # For each room
        for idx, room in enumerate(roomList):
            room.name = self.room_names[idx]

            topX, topY = room.top
            sizeX, sizeY = room.size

            # Draw the top and bottom walls
            for i in range(0, sizeX):
                self.grid.set(topX + i, topY, wall)
                self.grid.set(topX + i, topY + sizeY - 1, wall)

            # Draw the left and right walls
            for j in range(0, sizeY):
                self.grid.set(topX, topY + j, wall)
                self.grid.set(topX + sizeX - 1, topY + j, wall)

            # If this isn't the first room, place the entry door
            if idx > 0:
                # Pick a door color different from the previous one
                doorColors = set(COLOR_NAMES)
                if prevDoorColor:
                    doorColors.remove(prevDoorColor)
                # Note: the use of sorting here guarantees determinism,
                # This is needed because Python's set is not deterministic
                doorColor = self._rand_elem(sorted(doorColors))

                entryDoor = Door(doorColor)
                self.grid.set(*room.entryDoorPos, entryDoor)
                prevDoorColor = doorColor

                prevRoom = roomList[idx - 1]
                prevRoom.exitDoorPos = room.entryDoorPos

        # Randomize the starting agent position and direction
        self.place_agent(roomList[0].top, roomList[0].size)

        # Create rooms dict
        rooms_dict = {}
        for r in self.rooms:
            if r.name not in rooms_dict:
                rooms_dict[r.name] = {}
                rooms_dict[r.name]["P"] = {}
            rooms_dict[r.name]["T"] = r.temperature
            rooms_dict[r.name]["A"] = (r.size[0] - 2) * (r.size[1] - 2)
            rooms_dict[r.name]["heat"] = 0
            rooms_dict[r.name]["mask"] = np.zeros((self.width, self.height))
            rooms_dict[r.name]["mask"][r.y1 + 1:r.y2 - 1, r.x1 + 1:r.x2 - 1] = 1
            P_out = 2 * (r.x2 - r.x1 + r.y2 - r.y1 - 4)
            for r2 in self.rooms:
                if r2 == r:
                    pass
                else:
                    P = 0
                    if r.x1 + 1 == r2.x2 or r.x2 == r2.x1 + 1:
                        overlap = min(r2.y2 - 1, r.y2 - 1) - max(r2.y1,
                                                                 r.y1) - 1
                        if overlap > 0:
                            P += overlap
                            P_out -= overlap
                    if r.y1 + 1 == r2.y2 or r.y2 == r2.y1 + 1:
                        overlap = min(r2.x2 - 1, r.x2 - 1) - max(r2.x1,
                                                                 r.x1) - 1
                        if overlap > 0:
                            P += overlap
                            P_out -= overlap
                    rooms_dict[r.name]["P"][r2.name] = P
            rooms_dict[r.name]["P_out"] = P_out
            rooms_dict[r.name]["T_out"] = self.t_out
        self.rooms_dict = rooms_dict

        # Place the heating tiles in the house
        self.temperatures = np.ones((self.width, self.height)) * self.t_out
        for r, data in self.rooms_dict.items():
            self.temperatures[data["mask"] == 1] = data["T"]

        for i, cell in enumerate(self.grid.grid):
            x, y = divmod(i, width)

            if cell is None:
                self.grid.grid[i] = HeatingTile(
                    temperature=self.temperatures[x, y]
                )
            self.grid.grid[i].temperature = self.temperatures[x, y]

        # # Place the final goal in the last room
        # self.goal_pos = self.place_obj(Goal(), roomList[-1].top,
        #                                roomList[-1].size)

        self.mission = 'save the world'

    def _placeRoom(
            self,
            numLeft,
            roomList,
            minSz,
            maxSz,
            entryDoorWall,
            entryDoorPos
    ):
        # Choose the room size randomly
        sizeX = self._rand_int(minSz, maxSz + 1)
        sizeY = self._rand_int(minSz, maxSz + 1)

        # The first room will be at the door position
        if len(roomList) == 0:
            topX, topY = entryDoorPos
        # Entry on the right
        elif entryDoorWall == 0:
            topX = entryDoorPos[0] - sizeX + 1
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the south
        elif entryDoorWall == 1:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1] - sizeY + 1
        # Entry wall on the left
        elif entryDoorWall == 2:
            topX = entryDoorPos[0]
            y = entryDoorPos[1]
            topY = self._rand_int(y - sizeY + 2, y)
        # Entry wall on the top
        elif entryDoorWall == 3:
            x = entryDoorPos[0]
            topX = self._rand_int(x - sizeX + 2, x)
            topY = entryDoorPos[1]
        else:
            assert False, entryDoorWall

        # If the room is out of the grid, can't place a room here
        if topX < 0 or topY < 0:
            return False
        if topX + sizeX > self.width or topY + sizeY >= self.height:
            return False

        # If the room intersects with previous rooms, can't place it here
        for room in roomList[:-1]:
            nonOverlap = \
                topX + sizeX < room.top[0] or \
                room.top[0] + room.size[0] <= topX or \
                topY + sizeY < room.top[1] or \
                room.top[1] + room.size[1] <= topY

            if not nonOverlap:
                return False

        # Add this room to the list
        roomList.append(Room(
            (topX, topY),
            (sizeX, sizeY),
            entryDoorPos,
            None,
            "tmp_room_name",
            self.t_start,
        ))

        # If this was the last room, stop
        if numLeft == 1:
            return True

        # Try placing the next room
        for i in range(0, 8):

            # Pick which wall to place the out door on
            wallSet = set((0, 1, 2, 3))
            wallSet.remove(entryDoorWall)
            exitDoorWall = self._rand_elem(sorted(wallSet))
            nextEntryWall = (exitDoorWall + 2) % 4

            # Pick the exit door position
            # Exit on right wall
            if exitDoorWall == 0:
                exitDoorPos = (
                    topX + sizeX - 1,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on south wall
            elif exitDoorWall == 1:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY + sizeY - 1
                )
            # Exit on left wall
            elif exitDoorWall == 2:
                exitDoorPos = (
                    topX,
                    topY + self._rand_int(1, sizeY - 1)
                )
            # Exit on north wall
            elif exitDoorWall == 3:
                exitDoorPos = (
                    topX + self._rand_int(1, sizeX - 1),
                    topY
                )
            else:
                assert False

            # Recursively create the other rooms
            success = self._placeRoom(
                numLeft - 1,
                roomList=roomList,
                minSz=minSz,
                maxSz=maxSz,
                entryDoorWall=nextEntryWall,
                entryDoorPos=exitDoorPos
            )

            if success:
                break

        return True

    def _change_temperature(self, rooms_to_heat: List[int]):
        """ Changes the temperature of each object in the house """
        for r in rooms_to_heat:
            self.rooms_dict[self.room_names[r]]["heat"] = 1

        self.rooms_dict = self.model.step(self.rooms_dict)

        for r, data in self.rooms_dict.items():
            self.temperatures[data["mask"] == 1] = data["T"]

        for i, cell in enumerate(self.grid.grid):
            x, y = divmod(i, self.width)

            if cell is None:
                self.grid.grid[i] = HeatingTile(
                    temperature=self.temperatures[x, y]
                )
            self.grid.grid[i].temperature = self.temperatures[x, y]

    def step(self, action):
        self.current_dt += self.timedelta
        self.step_count += 1

        reward = 0
        done = False

        # Move each homie and determine their preference for the temperature
        homie_info = dict()
        for homie in self.homies:
            homie_info[homie] = {}
            homie_info[homie]["room"] = homie.current_room
            homie_info[homie]["dt"] = self.current_dt
            if homie.current_room == "Outside":
                homie_info[homie]["temperature"] = self.t_out
            else:
                homie_info[homie]["temperature"] = self.rooms_dict[
                    homie.current_room]["T"]
            homie_info[homie]["comfort"] = homie.get_preferred_temperature(
                self.current_dt)
            if not homie_info[homie]["comfort"][0] <= \
                   homie_info[homie]["temperature"] <= \
                   homie_info[homie]["comfort"][1]:
                reward += -(min(abs(homie_info[homie]["temperature"]
                                    - homie_info[homie]["comfort"][0]),
                                abs(homie_info[homie]["temperature"]
                                    - homie_info[homie]["comfort"][1]))
                            * self.homie_reward_scaler)

            homie.step(timestamp=self.current_dt)

        # Adjust the temperature in the house wrt to the preferences of homies
        self._change_temperature(action)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        # Remove the agent from the observation
        obs['image'][:, :, 0][obs['image'][:, :, 0] == 10] = 1

        return obs, reward, done, homie_info


if __name__ == '__main__':
    env = MultiRoomHouse(homies=[Homie(initial_room='Bedroom')])
    for _ in range(100):
        env.render(temperature=True)
        time.sleep(1)
    print(1)