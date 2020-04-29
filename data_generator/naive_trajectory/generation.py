import collections
import typing
import numpy as np

from matplotlib import pyplot as plt


LOCATION_BIAS = 5
HEIGHT_BIAS = 0.5
WITH_NOISE = True

DISPLAY_TAIL = 20
USE_MULTICOLOR = True


class GCSLocation(collections.namedtuple('GCSLocation', ('longitude', 'latitude', 'altitude'))):
    def __add__(self, other):
        if isinstance(other, GCSLocation):
            return GCSLocation(
                self.longitude + other.longitude,
                self.latitude + other.latitude,
                self.altitude + other.altitude)
        if isinstance(other, (typing.Sized, typing.Sequence)) and len(other) == 3:
            return self + GCSLocation(*tuple(other))
        raise ValueError('Unexpected value to add: {}.'.format(other))

    def __sub__(self, other):
        if isinstance(other, GCSLocation):
            return GCSLocation(
                self.longitude - other.longitude,
                self.latitude - other.latitude,
                self.altitude - other.altitude)
        if isinstance(other, (typing.Sized, typing.Sequence)) and len(other) == 3:
            return self - GCSLocation(*tuple(other))
        raise ValueError('Unexpected value to subtract: {}.'.format(other))

    @property
    def as_tuple(self):
        return tuple(self)

    def compute_distance(self, other):
        d = self - other
        return np.sqrt(np.square(d.longitude) + np.square(d.latitude) + np.square(d.altitude))


class Airspace(collections.namedtuple('Airspace', ('centre', 'radius'))):
    def generate_airline(self, distance, angle, height, clockwise=False):
        phy = np.arccos(distance / self.radius)
        location_1 = GCSLocation(
            self.centre.longitude + self.radius * np.cos(angle - phy),
            self.centre.latitude + self.radius * np.sin(angle - phy),
            height)
        location_2 = GCSLocation(
            self.centre.longitude + self.radius * np.cos(angle + phy),
            self.centre.latitude + self.radius * np.sin(angle + phy),
            height)
        if clockwise:
            return location_2, location_1
        return location_1, location_2

    def generate_trajectory(self,
                            start_location, end_location,
                            velocity,
                            start_time=0, dt=1,
                            location_bias=LOCATION_BIAS,
                            height_bias=HEIGHT_BIAS,
                            with_noise=WITH_NOISE):
        d = end_location - start_location
        rou = start_location.compute_distance(end_location)
        phy = np.arccos(d.altitude / rou)
        theta = np.arccos(d.longitude / rou / np.sin(phy))

        bias_theta = theta + 0.5 * np.pi
        bias_d = np.random.uniform(-location_bias, location_bias, size=(2,))
        bias_h = np.random.uniform(-height_bias, height_bias, size=(2,))
        start_location = start_location + (bias_d[0] * np.cos(bias_theta), bias_d[0] * np.sin(bias_theta), bias_h[0])
        end_location = end_location + (bias_d[1] * np.cos(bias_theta), bias_d[1] * np.sin(bias_theta), bias_h[1])

        d = end_location - start_location
        distance = d.compute_distance((0, 0, 0))

        dx = d.longitude / distance
        dy = d.latitude / distance
        dz = d.altitude / distance
        time = distance / velocity

        t = np.arange(start_time, start_time + time, dt)
        displacement = velocity * (t - start_time)
        longitude = start_location.longitude + displacement * dx
        latitude = start_location.latitude + displacement * dy
        altitude = start_location.altitude + displacement * dz

        if with_noise:
            r = np.random.normal(0., 0.001 * self.radius, size=np.shape(t))
            theta = np.random.normal(np.arctan(d.latitude / d.longitude) + 0.5 * np.pi, 0.01, size=np.shape(t))
            longitude = longitude + r * np.cos(theta)
            latitude = latitude + r * np.sin(theta)
            altitude = altitude

        return t, longitude, latitude, altitude

    @property
    def space_as_dots(self):
        theta = np.linspace(0, 2 * np.pi, 1000)
        x = self.centre.longitude + self.radius * np.cos(theta)
        y = self.centre.longitude + self.radius * np.sin(theta)
        return x, y


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
                distance=np.random.uniform(alert_radius, kwargs['radius']),
                angle=np.random.uniform(0, 2 * np.pi),
                height=np.random.uniform(min_height, max_height),
                clockwise=np.random.uniform() > 0.5))
        cls._normal_alert_airlines = []
        for i in range(num_normal_alert_airlines):
            cls._normal_alert_airlines.append(cls.generate_airline(
                distance=np.random.uniform(dangerous_radius, alert_radius),
                angle=np.random.uniform(0, 2 * np.pi),
                height=np.random.uniform(alert_height, max_height),
                clockwise=np.random.uniform() > 0.5))
        cls._anomaly_alert_airlines = []
        for i in range(num_anomaly_alert_airlines):
            cls._anomaly_alert_airlines.append(cls.generate_airline(
                distance=np.random.uniform(dangerous_radius, alert_radius),
                angle=np.random.uniform(0, 2 * np.pi),
                height=np.random.uniform(min_height, alert_height),
                clockwise=np.random.uniform() > 0.5))
        cls._anomaly_dangerous_airlines = []
        for i in range(num_anomaly_dangerous_airlines):
            cls._anomaly_dangerous_airlines.append(cls.generate_airline(
                distance=np.random.uniform(0, dangerous_radius),
                angle=np.random.uniform(0, 2 * np.pi),
                height=np.random.uniform(min_height, max_height),
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


def main():
    region = DetectingRegion(
        centre=GCSLocation(0, 0, 0),
        radius=200,
        dangerous_radius=50,
        alert_radius=100,
        max_height=15,
        alert_height=8,
        min_height=5,
        num_safe_airlines=10,
        num_normal_alert_airlines=10,
        num_anomaly_alert_airlines=1,
        num_anomaly_dangerous_airlines=1,
        num_anomaly_curve_airlines=10)
    (safe_trajectories, normal_alert_trajectories,
     anomaly_alert_trajectories, anomaly_dangerous_trajectories,
     anomaly_too_slow_trajectories, anomaly_too_fast_trajectories) = region.generate_trajectories(
        num_each_safe=2,
        num_each_normal_alert=2,
        num_each_anomaly_alert=0,
        num_each_anomaly_dangerous=1,
        num_each_anomaly_curve=1,
        num_each_anomaly_too_slow=0,
        num_each_anomaly_too_fast=1,
        min_velocity=0.1,
        alert_velocity_to_min=0.2,
        alert_velocity_to_max=0.3,
        max_velocity=1.0)

    print(len(anomaly_too_fast_trajectories), len(anomaly_too_slow_trajectories))
    plt.rcParams['figure.figsize'] = [7, 7]
    for frame in range(10000):
        plt.cla()
        plt.plot(*Airspace(GCSLocation(0, 0, 0), 50).space_as_dots, 'r,', color='grey')
        plt.plot(*Airspace(GCSLocation(0, 0, 0), 100).space_as_dots, 'r,', color='grey')
        plt.plot(*region.space_as_dots, 'r,', color='grey')
        for airline in region._safe_airlines + region._normal_alert_airlines:
            plt.plot(
                [airline[0].longitude, airline[1].longitude],
                [airline[0].latitude, airline[1].latitude],
                'r:', color='grey')
        for trajectory in safe_trajectories:
            t, longitude, latitude, altitude = trajectory
            plt.plot(
                longitude[max(0, frame - DISPLAY_TAIL):frame],
                latitude[max(0, frame - DISPLAY_TAIL):frame],
                'r-', color='green' if USE_MULTICOLOR else 'green')
        for trajectory in normal_alert_trajectories:
            t, longitude, latitude, altitude = trajectory
            plt.plot(
                longitude[max(0, frame - DISPLAY_TAIL):frame],
                latitude[max(0, frame - DISPLAY_TAIL):frame],
                'r-', color='blue' if USE_MULTICOLOR else 'green')
        for trajectory in anomaly_alert_trajectories:
            t, longitude, latitude, altitude = trajectory
            plt.plot(
                longitude[max(0, frame - DISPLAY_TAIL):frame],
                latitude[max(0, frame - DISPLAY_TAIL):frame],
                'r-', color='yellow' if USE_MULTICOLOR else 'red')
        for trajectory in anomaly_dangerous_trajectories:
            t, longitude, latitude, altitude = trajectory
            plt.plot(
                longitude[max(0, frame - DISPLAY_TAIL):frame],
                latitude[max(0, frame - DISPLAY_TAIL):frame],
                'r-', color='red' if USE_MULTICOLOR else 'red')
        for trajectory in anomaly_too_slow_trajectories:
            t, longitude, latitude, altitude = trajectory
            plt.plot(
                longitude[max(0, frame - DISPLAY_TAIL):frame],
                latitude[max(0, frame - DISPLAY_TAIL):frame],
                'r-', color='purple' if USE_MULTICOLOR else 'red')
        for trajectory in anomaly_too_fast_trajectories:
            t, longitude, latitude, altitude = trajectory
            plt.plot(
                longitude[max(0, frame - DISPLAY_TAIL):frame],
                latitude[max(0, frame - DISPLAY_TAIL):frame],
                'r-', color='orange' if USE_MULTICOLOR else 'red')

        plt.pause(0.001)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()







