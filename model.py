import numpy as np
from typing import Dict


class SimpleModel:
    def __init__(self,
                 RSI: float = 4.2,
                 wall_height: float = 3,  # m
                 wall_width: float = 10,  # m
                 air_capacity: float = 1005,  # J/(kg deg)
                 air_density: float = 1.204,  # kg / m3
                 timestep: int = 60,  # s
                 heater_output: float = 1000,  # W
                 round: int = 2,
                 ):
        """
        source for RSI:
        https://www.nrcan.gc.ca/sites/www.nrcan.gc.ca/files/energy/pdf/housing/Keeping-the-Heat-In_e.pdf
        (p15, Table 2-1)
        Good explanation of RSI:
        https://dothemath.ucsd.edu/2012/11/this-thermal-house/
        """
        wall_area = wall_width * wall_height
        self.A_over_RSI = wall_area / RSI
        room_mass = wall_area * wall_width * air_density
        self.air_capacity = air_capacity * room_mass
        assert (timestep <= 60), "timestep should be <= 60s to better " \
                                 "approximate heat transfer..."
        self.timestep = timestep
        self.round = round
        # assume heater running at half capacity
        self.heater_output = heater_output * self.timestep

    def step(self, temperatures: np.ndarray, heat: np.ndarray):
        dT_y = temperatures[:, 1:] - temperatures[:, :-1]
        dT_x = temperatures[1:, :] - temperatures[:-1, :]

        heat_flow_y = dT_y * self.A_over_RSI * self.timestep
        heat_flow_x = dT_x * self.A_over_RSI * self.timestep

        assert ((heat == 1) | (heat == 0)).all(), "Heat must be 1 or 0"
        heat *= self.heater_output
        heat[:, 1:] -= heat_flow_y
        heat[1:, :] -= heat_flow_x
        heat[:, :-1] += heat_flow_y
        heat[:-1, :] += heat_flow_x

        temperatures[1:-1, 1:-1] += heat[1:-1, 1:-1] / self.air_capacity
        heat *= 0
        return np.around(temperatures, self.round)


class RoomModel:
    def __init__(self,
                 RSI: float = 4.2 * 50,
                 wall_height: float = 2.5,  # m
                 wall_width: float = 1,  # m
                 air_capacity: float = 1005,  # J/(kg deg)
                 air_density: float = 1.204,  # kg / m3
                 timestep: int = 60,  # s
                 round: int = 2,
                 heat_per_m2: float = 200.
                 ):
        """
        source for RSI:
        https://www.nrcan.gc.ca/sites/www.nrcan.gc.ca/files/energy/pdf/housing/Keeping-the-Heat-In_e.pdf
        (p15, Table 2-1)
        Good explanation of RSI:
        https://dothemath.ucsd.edu/2012/11/this-thermal-house/
        """
        wall_area = wall_width * wall_height
        self.A_over_RSI = wall_area / RSI
        room_mass = wall_area * wall_width * air_density
        self.air_capacity = air_capacity * room_mass
        assert (timestep <= 60), "timestep should be <= 60s to better " \
                                 "approximate heat transfer..."
        self.timestep = timestep
        self.round = round
        self.heat_per_m2 = heat_per_m2

    def step(self, rooms_data: Dict):
        # calculate heat flow Q into a room from an adjacent room, given:
        #   T_room: room temperature
        #   T_adj: adjacent room temperature
        #   P: shared perimeter between adjacent rooms
        for room, data in rooms_data.items():
            T_room = data["T"]
            data["Q"] = 0
            for adj_room, P in data["P"].items():
                T_out = rooms_data[adj_room]["T"]
                data["Q"] += P * (T_out - T_room) * self.A_over_RSI * self.timestep
            data["Q"] += data["P_out"] * (data["T_out"] - T_room) * self.A_over_RSI * self.timestep
            if data["heat"] == 1:
                data["Q"] += self.heat_per_m2 * data["A"]
                data["heat"] = 0

        for room, data in rooms_data.items():
            print(data["Q"], data["A"])
            data["T"] += data["Q"] / self.air_capacity
            data["T"] = np.round(data["T"], 2)

        return rooms_data


if __name__ == '__main__':
    house_shape = (10, 5)
    T_start = 25
    T_out = 0

    wall_mask = np.ones(house_shape)
    wall_mask[1:-1, 1:-1] = 0

    temperatures = np.ones(house_shape) * T_start
    temperatures[wall_mask == 1] = T_out

    heat = np.zeros(house_shape)

    sm = SimpleModel()
    for _ in range(1000):
        print(temperatures)
        temperatures = sm.step(temperatures, heat)