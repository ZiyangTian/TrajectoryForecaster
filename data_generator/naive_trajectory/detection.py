import collections
import os
import typing
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as img


LOCATION_BIAS = 5
HEIGHT_BIAS = 0.5
WITH_NOISE = True

DISPLAY_TAIL = 20
USE_MULTICOLOR = True


MAX_VELOCITY = 1
MIN_VELOCITY = 0.1
MAX_RADIUS = 200
MAX_HEIGHT = 12
MIN_HEIGHT = 6

DANGEROUS_RADIUS = 50
ALERT_RADIUS = 100
ALERT_HEIGHT = 8
MAX_NORMAL_VELOCITY = 0.3
MIN_NORMAL_VELOCITY = 0.2

DT = 5


class Point(collections.namedtuple('GCSLocation', ('x', 'y', 'z'))):
    def __add__(self, other):
        if isinstance(other, Point):
            return Point(
                self.x + other.x,
                self.y + other.y,
                self.z + other.z)
        if isinstance(other, (typing.Sized, typing.Sequence)) and len(other) == 3:
            return self + Point(*tuple(other))
        raise ValueError('Unexpected value to add: {}.'.format(other))

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z)
        if isinstance(other, (typing.Sized, typing.Sequence)) and len(other) == 3:
            return self - Point(*tuple(other))
        raise ValueError('Unexpected value to subtract: {}.'.format(other))

    @property
    def distance_to_origin(self):
        return np.sqrt(np.square(self.x) + np.square(self.y) + np.square(self.z))

    def compute_distance(self, other):
        return (self - other).distance_to_origin


def segment(airline, airspace):
    phy = np.arccos(airline.d / airspace.R)
    location_1 = Point(
        airspace.x_loc + airspace.R * np.cos(airline.alpha - phy),
        airspace.y_loc + airspace.R * np.sin(airline.alpha - phy),
        airline.h)
    location_2 = Point(
        airspace.x_loc + airspace.R * np.cos(airline.alpha + phy),
        airspace.y_loc + airspace.R * np.sin(airline.alpha + phy),
        airline.h)
    if airline.clockwise:
        return location_2, location_1
    return location_1, location_2


class Airline(collections.namedtuple('Airline', ('d', 'alpha', 'h', 'clockwise'))):
    def plot(self, airspace):
        location_1, location_2 = segment(self, airspace)
        plt.plot([location_1.x, location_2.x], [location_1.y, location_2.y], ':', color='grey')


class DetectedTrajectory(object):
    """
    t, x, y, z,
    v, direction
    anomaly_by_overspeed,
    anomaly_by_underspeed,
    anomaly_by_too_close,
    anomaly_by_low_altitude
    """
    def __init__(self, t, x, y, z, airline: Airline = None):
        self._data = pd.DataFrame(np.array([t, x, y, z]).T, columns=['t', 'x', 'y', 'z'])
        self._airline = airline
        self.compute_velocity()
        self.compute_anomaly()

    @property
    def data(self):
        return self._data

    def compute_velocity(self):
        dt = np.array(self._data['t'][1:]) - np.array(self._data['t'][:-1])
        dx = np.array(self._data['x'][1:]) - np.array(self._data['x'][:-1])
        dy = np.array(self._data['y'][1:]) - np.array(self._data['y'][:-1])
        dz = np.array(self._data['z'][1:]) - np.array(self._data['z'][:-1])
        if 'v' not in self._data:
            v = Point(dx, dy, dz).distance_to_origin / dt
            v = np.concatenate([np.array([0]), v])
            self._data['v'] = v
        if 'direction' not in self._data:
            direction = np.concatenate([np.array([0]), dy / dx])
            self._data['direction'] = direction

    def compute_anomaly(self):
        def truncate_to_01(x):
            x = np.where(x < 0, np.broadcast_to(0, np.shape(x)), x)
            x = np.where(x > 1, np.broadcast_to(1, np.shape(x)), x)
            return x

        self._data['anomaly_by_overspeed'] = truncate_to_01(
            (self._data['v'] - MAX_NORMAL_VELOCITY) / (MAX_VELOCITY - MAX_NORMAL_VELOCITY))
        self._data['anomaly_by_underspeed'] = truncate_to_01(
            (MIN_NORMAL_VELOCITY - self._data['v']) / (MIN_NORMAL_VELOCITY - MIN_VELOCITY))
        if self._airline is None:
            self._data['anomaly_by_too_close'] = truncate_to_01(
                (DANGEROUS_RADIUS - Point(self._data['x'], self._data['y'], self._data['z']).distance_to_origin) / DANGEROUS_RADIUS)
        else:
            self._data['anomaly_by_too_close'] = truncate_to_01(
                np.broadcast_to((DANGEROUS_RADIUS - self._airline.d) / DANGEROUS_RADIUS, np.shape(self._data['t'])))
        self._data['anomaly_by_low_altitude'] = truncate_to_01(
            (ALERT_HEIGHT - self._data['z']) / (ALERT_HEIGHT - MIN_HEIGHT))
        if self._airline is None:
            self._data['anomaly_by_low_altitude'] = np.where(
                Point(self._data['x'], self._data['y'], 0).distance_to_origin > ALERT_RADIUS,
                np.zeros_like(self._data['anomaly_by_low_altitude']),
                self._data['anomaly_by_low_altitude'])
        else:
            if self._airline.d > ALERT_RADIUS:
                self._data['anomaly_by_low_altitude'] = 0

    def plot(self, *args, **kwargs):
        plt.plot(self._data['x'], self._data['y'], *args, **kwargs)


class Airspace(collections.namedtuple('Airspace', ('x_loc', 'y_loc', 'R'))):
    def airline_segment(self, airline: Airline):
        phy = np.arccos(airline.d / self.R)
        location_1 = Point(
            self.x_loc + self.R * np.cos(airline.alpha - phy),
            self.y_loc + self.R * np.sin(airline.alpha - phy),
            airline.h)
        location_2 = Point(
            self.x_loc + self.R * np.cos(airline.alpha + phy),
            self.y_loc + self.R * np.sin(airline.alpha + phy),
            airline.h)
        if airline.clockwise:
            return location_2, location_1
        return location_1, location_2

    def generate_trajectory(self,
                            airline: Airline,
                            velocity,
                            start_time=0, dt=1,
                            location_bias=LOCATION_BIAS,
                            height_bias=HEIGHT_BIAS,
                            with_noise=WITH_NOISE,
                            save_as=None):
        start_location, end_location = self.airline_segment(airline)

        distance = end_location - start_location
        rou = start_location.compute_distance(end_location)
        phy = np.arccos(distance.z / rou)
        theta = np.arccos(distance.x / rou / np.sin(phy))

        bias_theta = theta + 0.5 * np.pi
        bias_d = np.random.uniform(-location_bias, location_bias, size=(2,))
        bias_h = np.random.uniform(-height_bias, height_bias, size=(2,))
        start_location = start_location + (bias_d[0] * np.cos(bias_theta), bias_d[0] * np.sin(bias_theta), bias_h[0])
        end_location = end_location + (bias_d[1] * np.cos(bias_theta), bias_d[1] * np.sin(bias_theta), bias_h[1])

        distance = end_location - start_location
        d = distance.compute_distance((0, 0, 0))

        dx = distance.x / d
        dy = distance.y / d
        dz = distance.z / d
        time = d / velocity

        t = np.arange(start_time, start_time + time, dt)
        displacement = velocity * (t - start_time)
        x = start_location.x + displacement * dx
        y = start_location.y + displacement * dy
        z = start_location.z + displacement * dz

        if with_noise:
            r = np.random.normal(0., 0.001 * self.R, size=np.shape(t))
            theta = np.random.normal(np.arctan(distance.y / distance.x) + 0.5 * np.pi, 0.01, size=np.shape(t))
            x = x + r * np.cos(theta)
            y = y + r * np.sin(theta)
            z = z
        trajectory = DetectedTrajectory(t, x, y, z, airline)
        if save_as is not None:
            trajectory.data.to_csv(save_as, index=False)

        return trajectory

    @property
    def as_dots(self):
        theta = np.linspace(0, 2 * np.pi, 1000)
        x = self.x_loc + self.R * np.cos(theta)
        y = self.y_loc + self.R * np.sin(theta)
        return x, y

    def plot(self):
        plt.plot(*Airspace(0, 0, 50).as_dots, 'r-.', color='red')
        plt.plot(*Airspace(0, 0, 100).as_dots, 'r-.', color='orange')
        plt.plot(*self.as_dots, '-', color='grey')


class DetectingRegion(Airspace):
    def __new__(cls, **kwargs):
        dangerous_radius = kwargs.pop('dangerous_radius')
        alert_radius = kwargs.pop('alert_radius')
        max_height = kwargs.pop('max_height')
        alert_height = kwargs.pop('alert_height')
        min_height = kwargs.pop('min_height')
        num_safe_airlines = kwargs.pop('num_safe_airlines')
        num_normal_alert_airlines = kwargs.pop('num_normal_alert_airlines')
        num_anomaly_alert_airlines = kwargs.pop('num_anomaly_alert_airlines')
        num_anomaly_dangerous_airlines = kwargs.pop('num_anomaly_dangerous_airlines')
        num_anomaly_curve_airlines = kwargs.pop('num_anomaly_curve_airlines')
        cls = super(DetectingRegion, cls).__new__(cls, **kwargs)

        cls._safe_airlines = []
        for i in range(num_safe_airlines):
            cls._safe_airlines.append(cls.generate_airline(
                d=np.random.uniform(alert_radius, kwargs['R']),
                alpha=np.random.uniform(0, 2 * np.pi),
                h=np.random.uniform(min_height, max_height),
                clockwise=np.random.uniform() > 0.5))
        cls._normal_alert_airlines = []
        for i in range(num_normal_alert_airlines):
            cls._normal_alert_airlines.append(cls.generate_airline(
                d=np.random.uniform(dangerous_radius, alert_radius),
                alpha=np.random.uniform(0, 2 * np.pi),
                h=np.random.uniform(alert_height, max_height),
                clockwise=np.random.uniform() > 0.5))
        cls._anomaly_alert_airlines = []
        for i in range(num_anomaly_alert_airlines):
            cls._anomaly_alert_airlines.append(cls.generate_airline(
                d=np.random.uniform(dangerous_radius, alert_radius),
                alpha=np.random.uniform(0, 2 * np.pi),
                h=np.random.uniform(min_height, alert_height),
                clockwise=np.random.uniform() > 0.5))
        cls._anomaly_dangerous_airlines = []
        for i in range(num_anomaly_dangerous_airlines):
            cls._anomaly_dangerous_airlines.append(cls.generate_airline(
                d=np.random.uniform(0, dangerous_radius),
                alpha=np.random.uniform(0, 2 * np.pi),
                h=np.random.uniform(min_height, max_height),
                clockwise=np.random.uniform() > 0.5))
        cls._anomaly_curve_airlines = []
        cls._all_airlines = cls._safe_airlines + cls._normal_alert_airlines + cls._anomaly_alert_airlines + \
                            cls._anomaly_dangerous_airlines + cls._anomaly_curve_airlines
        return cls

    def generate_trajectories(self,
                              num_each_safe,
                              num_each_normal_alert,
                              num_each_anomaly_alert,
                              num_each_anomaly_dangerous,
                              num_each_anomaly_curve,
                              num_each_anomaly_too_slow,
                              num_each_anomaly_too_fast,
                              min_velocity,
                              alert_velocity_to_min,
                              alert_velocity_to_max,
                              max_velocity):
        safe_trajectories = []
        for airline in self._safe_airlines:
            for i in range(num_each_safe):
                safe_trajectories.append(self.generate_trajectory(
                    airline[0], airline[1], velocity=np.random.uniform(alert_velocity_to_min, alert_velocity_to_max)))
        normal_alert_trajectories = []
        for airline in self._normal_alert_airlines:
            for i in range(num_each_normal_alert):
                normal_alert_trajectories.append(self.generate_trajectory(
                    airline[0], airline[1], velocity=np.random.uniform(alert_velocity_to_min, alert_velocity_to_max)))
        anomaly_alert_trajectories = []
        for airline in self._anomaly_alert_airlines:
            for i in range(num_each_anomaly_alert):
                anomaly_alert_trajectories.append(self.generate_trajectory(
                    airline[0], airline[1], velocity=np.random.uniform(alert_velocity_to_min, alert_velocity_to_max)))
        anomaly_dangerous_trajectories = []
        for airline in self._anomaly_dangerous_airlines:
            for i in range(num_each_anomaly_dangerous):
                anomaly_dangerous_trajectories.append(self.generate_trajectory(
                    airline[0], airline[1], velocity=np.random.uniform(alert_velocity_to_min, alert_velocity_to_max)))
        anomaly_too_fast_trajectories = []
        anomaly_too_slow_trajectories = []
        for airline in self._all_airlines:
            for i in range(num_each_anomaly_too_fast):
                anomaly_too_fast_trajectories.append(self.generate_trajectory(
                    airline[0], airline[1], velocity=np.random.uniform(alert_velocity_to_max, max_velocity)))
            for i in range(num_each_anomaly_too_slow):
                anomaly_too_slow_trajectories.append(self.generate_trajectory(
                    airline[0], airline[1], velocity=np.random.uniform(min_velocity, alert_velocity_to_min)))
        return (safe_trajectories, normal_alert_trajectories,
                anomaly_alert_trajectories, anomaly_dangerous_trajectories,
                anomaly_too_slow_trajectories, anomaly_too_fast_trajectories)


def generate_anomaly_demo(airspace: Airspace, d, alpha, r, h,
                          velocity,
                          start_time=0, dt=1,
                          with_noise=WITH_NOISE,
                          save_as=None):
    phy = np.arccos((airspace.R ** 2 + d ** 2 - r ** 2) / (2 * airspace.R * d))
    # if phy > 0.5 * np.pi:
    #     phy = np.pi - phy
    theta_1 = np.arcsin((airspace.R * np.sin(alpha + phy) - d * np.sin(alpha)) / r)
    theta_2 = np.arcsin((airspace.R * np.sin(alpha - phy) - d * np.sin(alpha)) / r)
    theta_1 = np.pi - theta_1
    if theta_2 < theta_1:
        theta_2 = np.pi * 2 + theta_2

    distance = r * (theta_2 - theta_1)
    time = distance / velocity
    angular_velocity = velocity / r

    t = np.arange(start_time, start_time + time, dt)
    theta = (t - start_time) * angular_velocity + theta_1
    x = airspace.x_loc + d * np.cos(alpha) + r * np.cos(theta)
    y = airspace.y_loc + d * np.sin(alpha) + r * np.sin(theta)
    z = np.broadcast_to(h, np.shape(t))

    if with_noise:
        r = np.random.normal(0., 0.001 * airspace.R, size=np.shape(t))
        phy = np.random.normal(0, 2 * np.pi, size=np.shape(t))
        x = x + r * np.cos(phy)
        y = y + r * np.sin(phy)
        z = z
    trajectory = DetectedTrajectory(t, x, y, z, None)
    if save_as is not None:
        trajectory.data.to_csv(save_as, index=False)
    return trajectory


def main():
    data_dir = '/Users/Tianziyang/Desktop/fake'
    num_safe_airlines = 10
    num_normal_airlines = 2
    num_normal_each_airline = 5

    airspace = Airspace(0, 0, MAX_RADIUS)

    # Generate airlines.
    safe_airlines = []
    for i in range(num_safe_airlines):
        safe_airlines.append(
            Airline(
                np.random.uniform(ALERT_RADIUS, MAX_RADIUS - 50),
                np.random.uniform(0, 2 * np.pi),
                # np.random.normal(1, 0.1),
                np.random.uniform(MIN_HEIGHT, MAX_HEIGHT),
                np.random.uniform() > 0.5))
    normal_airlines = []
    for i in range(num_normal_airlines):
        normal_airlines.append(
            Airline(
                np.random.uniform(DANGEROUS_RADIUS, ALERT_RADIUS),
                np.random.uniform(0, 2 * np.pi),
                # np.random.normal(1, 0.1),
                np.random.uniform(ALERT_HEIGHT, MAX_HEIGHT),
                np.random.uniform() > 0.5))
    anomaly_airline = Airline(
        np.random.uniform(0, ALERT_RADIUS),
        np.random.uniform(0, 2 * np.pi),
        np.random.uniform(MIN_HEIGHT, MAX_HEIGHT),
        np.random.uniform() > 0.5)

    # Generate trajectories.
    normal_trajectories = []
    for j, airline in enumerate(safe_airlines):
        trajectories = []
        for i in range(num_normal_each_airline):
            trajectories.append(airspace.generate_trajectory(
                airline, np.random.uniform(MIN_NORMAL_VELOCITY, MAX_NORMAL_VELOCITY),
                dt=DT,
                save_as=os.path.join(data_dir, 'safe_{}_{}.csv'.format(i, j))).data)
        normal_trajectories.append(pd.concat(trajectories))
    for j, airline in enumerate(normal_airlines):
        trajectories = []
        for i in range(num_normal_each_airline):
            trajectories.append(
                airspace.generate_trajectory(
                    airline, np.random.uniform(MIN_NORMAL_VELOCITY, MAX_NORMAL_VELOCITY),
                    dt=DT,
                    save_as=os.path.join(data_dir, 'normal_{}_{}.csv'.format(i, j))).data)
        normal_trajectories.append(pd.concat(trajectories))

    anomaly_trajectories = [
        airspace.generate_trajectory(
            anomaly_airline,
            np.random.uniform(MAX_NORMAL_VELOCITY, 0.4),
            dt=DT,
            save_as=os.path.join(data_dir, 'anomaly_1.csv')),
        airspace.generate_trajectory(
            normal_airlines[0],
            np.random.uniform(MIN_VELOCITY, MIN_NORMAL_VELOCITY),
            dt=DT,
            save_as=os.path.join(data_dir, 'anomaly_2.csv')),
        generate_anomaly_demo(
            airspace, 160, np.pi / 6, 100, 7, 0.25,
            dt=DT,
            save_as=os.path.join(data_dir, 'anomaly_3.csv')),
        airspace.generate_trajectory(
            Airline(30, np.random.uniform(0, 2 * np.pi), 10, np.random.uniform() > 0.5),
            0.25, dt=DT, save_as=os.path.join(data_dir, 'anomaly_4.csv'))
    ]
    # anomaly_trajectories = pd.concat(list(map(lambda t: t.data, anomaly_trajectories)))

    plt.rcParams['figure.figsize'] = [7.2, 7.2]
    # airspace.plot()
    # for tra in normal_trajectories + anomaly_trajectories:
    #     tra.plot()
    # #anomaly_trajectories[-1].plot()
    # plt.show()

    bgimg = img.imread('/Users/Tianziyang/Desktop/1.png')

    for frame in range(10, 200):
        plt.cla()
        plt.xlim([-220, 220])
        plt.ylim([-220, 220])
        airspace.plot()
        for airline in safe_airlines + normal_airlines:
            airline.plot(airspace)
        for i, trajectory in enumerate(normal_trajectories):
            plt.plot(
                trajectory['x'][max(0, frame - i * 5): max(0, frame-i * 5 + 1)],
                trajectory['y'][max(0, frame - i * 5): max(0, frame-i * 5 + 1)],
                'ro', color='green')
        plt.plot(
            anomaly_trajectories[0].data['x'][frame: frame + 1],
            anomaly_trajectories[0].data['y'][frame: frame + 1],
            'h', color='red')
        plt.plot(
            anomaly_trajectories[1].data['x'][frame: frame + 1],
            anomaly_trajectories[1].data['y'][frame: frame + 1],
            's', color='red')
        plt.plot(
            anomaly_trajectories[2].data['x'][frame: frame + 1],
            anomaly_trajectories[2].data['y'][frame: frame + 1],
            'D', color='red')
        plt.plot(
            anomaly_trajectories[3].data['x'][frame: frame + 1],
            anomaly_trajectories[3].data['y'][frame: frame + 1],
            '^', color='red')

        # 绘制表格
        ax = plt.gca()
        y = np.random.randn(9)
        col_labels = ['★', '●', '◆', '▲']
        row_labels = [
            'High-velocity Anomaly / %',
            'Low-velocity Anomaly / %',
            'Altitude Anomaly / %',
            'Close Anomaly / %']

        def formatted(index, key):
            try:
                return '{:.2f}'.format(100 * anomaly_trajectories[index].data[key][frame])
            except KeyError:
                return '0'

        table_vals = np.array([
            [
                formatted(i, 'anomaly_by_too_close'),
                formatted(i, 'anomaly_by_low_altitude'),
                formatted(i, 'anomaly_by_overspeed'),
                formatted(i, 'anomaly_by_underspeed')
            ] for i in range(len(anomaly_trajectories))
        ]).T
        pd.DataFrame(table_vals).to_csv(
            '/Users/Tianziyang/Desktop/fake/tables/{}.csv'.format(frame), index=False, header=False)


        # row_colors = np.where(table_vals.astype(np.float) > 0, 'red', 'white')

        # my_table = plt.table(
        #     cellText=table_vals, colWidths=[0.1] * 4, rowLabels=row_labels, colLabels=col_labels,
        #     cellColours=row_colors, loc='best', cellLoc='middle')
        # my_table.auto_set_font_size(False)
        # my_table.set_fontsize(7)
        # my_table.scale(1, 1)
        # plt.pause(0.01)
        plt.savefig('/Users/Tianziyang/Desktop/fake/images/{}.jpg'.format(frame), dpi=300)
    #plt.ioff()
    #plt.show()


if __name__ == '__main__':
    main()
