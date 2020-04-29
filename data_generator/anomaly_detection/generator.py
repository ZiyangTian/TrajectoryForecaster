import random
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_generator.anomaly_detection import environment
from data_generator.anomaly_detection import anomaly


class TrajectoryGeneratorConfig(object):
    # Global parameter scale.
    x_loc = 0
    y_loc = 0
    max_speed = 1
    min_speed = 0.1
    max_radius = 300
    max_height = 12
    min_height = 6

    # Anomaly scale.
    dangerous_radius = 50
    alert_radius = 100
    alert_height = 8
    max_normal_speed = 0.3
    min_normal_speed = 0.2

    # Data generation.
    dt = 1

    num_normal_safe_airlines = 8
    num_normal_alert_airlines = 2
    prob_normal_safe = 0.8
    prob_normal_alert = 0.1
    prob_anomaly_straight = 0.05
    prob_anomaly_circle = 0.025
    prob_anomaly_random = 0.025
    prob_distance_anomaly = 0.3
    prob_height_anomaly = 0.3
    prob_high_speed_anomaly = 0.6
    prob_low_speed_anomaly = 0.1
    prob_airline_anomaly = 0.6

    airline_noise_stddev = 5
    trajectory_noise = 0.1


class TrajectoryGenerator(TrajectoryGeneratorConfig):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)

        self._airspace = environment.Airspace(self.x_loc, self.y_loc, self.max_radius)

        self._normal_safe_airlines = []
        for _ in range(self.num_normal_safe_airlines):
            self._normal_safe_airlines.append(
                self._airspace.generate_straight_airline(
                    np.random.uniform(self.alert_radius, self.max_radius),
                    np.random.uniform(0, 2 * np.pi),
                    np.random.uniform(self.min_height, self.max_height),
                    clockwise=np.random.uniform() > 0.5))

        self._normal_alert_airlines = []
        for _ in range(self.num_normal_alert_airlines):
            self._normal_alert_airlines.append(
                self._airspace.generate_straight_airline(
                    np.random.uniform(self.dangerous_radius, self.alert_radius),
                    np.random.uniform(0, 2 * np.pi),
                    np.random.uniform(self.alert_height, self.max_height),
                    clockwise=np.random.uniform() > 0.5))

    @property
    def airspace(self):
        return self._airspace

    def generate_trajectory_from_airline(self, airline, speed):
        noise_r = np.random.normal(0, self.airline_noise_stddev, size=(2,))
        noise_theta = np.random.uniform(0, 2 * np.pi, size=(2,))
        noise_z = np.random.normal(0, 0.01, size=(2,))
        noise_x = noise_r * np.cos(noise_theta)
        noise_y = noise_r * np.sin(noise_theta)

        airline = environment.StraightAirline(
            airline.starting_location + (noise_x[0], noise_y[0], noise_z[0]),
            airline.ending_location + (noise_x[1], noise_y[1], noise_z[1]))

        initial_location = airline.starting_location
        displacement = airline.ending_location - airline.starting_location
        distance = displacement.modulus
        velocity = displacement / distance * speed
        time = distance / speed

        objective = environment.UniformLinearObjective(initial_location, velocity)
        objective.move_n(time, self.dt)
        trajectory = objective.get_noised(self.trajectory_noise)
        return trajectory

    def generate_normal_safe_trajectory(self):
        airline = random.choice(self._normal_safe_airlines)
        speed = np.random.uniform(self.min_normal_speed, self.max_normal_speed)
        trajectory = self.generate_trajectory_from_airline(airline, speed)
        for a in anomaly.ANOMALY:
            trajectory[a] = np.zeros((len(trajectory),), dtype=np.int)
        return trajectory

    def generate_normal_alert_trajectory(self):
        airline = random.choice(self._normal_alert_airlines)
        speed = np.random.uniform(self.min_normal_speed, self.max_normal_speed)

        trajectory = self.generate_trajectory_from_airline(airline, speed)
        for a in anomaly.ANOMALY:
            trajectory[a] = np.zeros((len(trajectory),), dtype=np.int)
        return trajectory

    def generate_anomaly_straight_trajectory(self):
        distance_anomaly = np.random.uniform() < self.prob_distance_anomaly
        height_anomaly = np.random.uniform() < self.prob_height_anomaly
        high_speed_anomaly = np.random.uniform() < self.prob_high_speed_anomaly
        low_speed_anomaly = np.random.uniform() < self.prob_low_speed_anomaly
        airline_anomaly = np.random.uniform() < self.prob_airline_anomaly

        if distance_anomaly:
            airline_anomaly = True
            distance = np.random.uniform(0, self.dangerous_radius)
            if height_anomaly:
                height = np.random.uniform(self.min_height, self.alert_height)
            else:
                height = np.random.uniform(self.alert_height, self.max_height)
            airline = self._airspace.generate_straight_airline(
                distance, np.random.uniform(0, 2 * np.pi), height, clockwise=np.random.uniform() > 0.5)
        elif height_anomaly:
            airline_anomaly = True
            distance = np.random.uniform(self.dangerous_radius, self.alert_radius)
            height = np.random.uniform(self.min_height, self.alert_height)
            airline = self._airspace.generate_straight_airline(
                distance, np.random.uniform(0, 2 * np.pi), height, clockwise=np.random.uniform() > 0.5)
        elif airline_anomaly:
            distance = np.random.uniform(self.alert_radius, self.max_radius)
            if distance > self.alert_radius:
                height = np.random.uniform(self.min_height, self.max_height)
            else:
                height = np.random.uniform(self.alert_height, self.max_height)
            airline = self._airspace.generate_straight_airline(
                distance, np.random.uniform(0, 2 * np.pi), height, clockwise=np.random.uniform() > 0.5)
        else:
            airline = random.choice(self._normal_alert_airlines + self._normal_safe_airlines)

        if high_speed_anomaly:
            speed = np.random.uniform(self.max_normal_speed, self.max_speed)
            low_speed_anomaly = False
        elif low_speed_anomaly:
            speed = np.random.uniform(self.min_speed, self.min_normal_speed)
        else:
            speed = np.random.uniform(self.min_normal_speed, self.max_normal_speed)

        trajectory = self.generate_trajectory_from_airline(airline, speed)

        def label(title):
            return np.ones((len(trajectory),), dtype=np.int) if title else np.zeros((len(trajectory),), dtype=np.int)

        trajectory['distance_anomaly'] = label(distance_anomaly)
        trajectory['distance_anomaly'][len(trajectory) // 2:] = 0
        trajectory['height_anomaly'] = label(height_anomaly)
        trajectory['high_speed_anomaly'] = label(high_speed_anomaly)
        trajectory['low_speed_anomaly'] = label(low_speed_anomaly)
        trajectory['airline_anomaly'] = label(airline_anomaly)
        return trajectory

    def generate_circular_trajectory(self, d, alpha, r, h, speed, clockwise=None):
        # phy = np.arccos((self._airspace.radius ** 2 + d ** 2 - r ** 2) / (2 * self._airspace.radius * d))
        phy = np.arccos((d ** 2 + r ** 2 - self._airspace.radius ** 2) / (2 * r * d))
        phy1 = alpha + np.pi - phy
        phy2 = alpha + np.pi + phy

        distance = r * (phy2 - phy1)
        time = distance / speed
        angular_velocity = speed / r
        t = np.arange(0, time, self.dt)

        if clockwise is None:
            clockwise = np.random.uniform() > 0.5
        if clockwise:
            theta = t * angular_velocity + phy1
        else:
            theta = phy2 - t * angular_velocity
        x = self._airspace.x_loc + d * np.cos(alpha) + r * np.cos(theta)
        y = self._airspace.y_loc + d * np.sin(alpha) + r * np.sin(theta)
        z = np.broadcast_to(h, np.shape(t))

        # compute velocity
        # dt = np.array(t[1:]) - np.array(t[:-1])
        dx = np.array(x[1:]) - np.array(x[:-1])
        dy = np.array(y[1:]) - np.array(y[:-1])
        # dz = np.array(z[1:]) - np.array(z[:-1])
        # v = environment.Velocity(dx, dy, dz).modulus / self.dt
        # v = np.concatenate([np.array([0]), v])
        direction = np.concatenate([np.array([0]), dy / dx])

        # compute anomaly
        distance = environment.StraightLine2D.from_vector(x, y, direction).distance_to_point(
            self._airspace.x_loc, self._airspace.y_loc)
        height = z

        distance_anomaly = np.less(distance, self.dangerous_radius).astype(np.int)
        height_anomaly = np.logical_and(
            np.logical_and(self.dangerous_radius < distance, distance < self.alert_radius),
            height < self.alert_height).astype(np.int)
        high_speed_anomaly = np.broadcast_to(speed > self.max_normal_speed, (len(t),)).astype(np.int)
        low_speed_anomaly = np.broadcast_to(speed < self.min_normal_speed, (len(t),)).astype(np.int)
        airline_anomaly = np.ones_like(t, dtype=np.int)

        # add noise.
        r = np.random.normal(0., 0.001 * self._airspace.radius, size=np.shape(t))
        phy = np.random.normal(0, 2 * np.pi, size=np.shape(t))
        x = x + r * np.cos(phy)
        y = y + r * np.sin(phy)
        z = z

        trajectory = pd.DataFrame({
            't': t, 'x': x, 'y': y, 'z': z,
            'distance_anomaly': distance_anomaly,
            'height_anomaly': height_anomaly,
            'high_speed_anomaly': high_speed_anomaly,
            'low_speed_anomaly': low_speed_anomaly,
            'airline_anomaly': airline_anomaly})
        trajectory['distance_anomaly'][len(trajectory) // 2:] = 0

        return trajectory

    def generate_random_trajectory(self, initial_speed, initial_height,
                                   acceleration_min, acceleration_max, height_stddev):
        initial_location_theta = np.random.uniform(0, 2 * np.pi)
        initial_location = environment.Location(
            self._airspace.radius * np.cos(initial_location_theta),
            self._airspace.radius * np.sin(initial_location_theta),
            initial_height)
        initial_velocity = environment.Location(
            self._airspace.x_loc, self._airspace.y_loc, initial_height) - initial_location
        initial_velocity = initial_velocity / initial_velocity.modulus * initial_speed
        objective = environment.RandomObjective(
            initial_location, initial_velocity,
            acceleration_min, acceleration_max, height_stddev,
            speed_min=self.min_speed, speed_max=self.max_speed)
        while (objective.location - (self._airspace.x_loc, self._airspace.y_loc, objective.location.z)).modulus \
                <= self._airspace.radius:
            objective.move(self.dt)
        trajectory = objective.get_noised(self.trajectory_noise)

        distance = environment.StraightLine2D.from_vector(
            np.array(objective.trajectory['x'], dtype=np.float),
            np.array(objective.trajectory['y'], dtype=np.float),
            np.array(
                objective.trajectory['vy'], dtype=np.float) / np.array(
                objective.trajectory['vx'], dtype=np.float)).distance_to_point(
                self._airspace.x_loc, self._airspace.y_loc)
        angle_cosine = (environment.Location(
            trajectory['x'],
            trajectory['y'],
            trajectory['z']) - (np.broadcast_to(self._airspace.x_loc, [len(trajectory)]),
                                np.broadcast_to(self._airspace.y_loc, [len(trajectory)]),
                                trajectory['z'])).dot((objective.trajectory['vx'],
                                                       objective.trajectory['vy'],
                                                       np.broadcast_to(0., [len(trajectory)]))).astype(np.int)
        height = objective.trajectory['z']
        speed = environment.Velocity(
            objective.trajectory['vx'],
            objective.trajectory['vy'],
            objective.trajectory['vz']).modulus
        trajectory['distance_anomaly'] = np.logical_and(
            np.less(distance, self.dangerous_radius).astype(np.int),
            np.greater_equal(angle_cosine, 0))
        trajectory['height_anomaly'] = np.logical_and(
            np.logical_and(self.dangerous_radius < distance, distance < self.alert_radius),
            height < self.alert_height).astype(np.int)

        trajectory['high_speed_anomaly'] = (speed > self.max_normal_speed).astype(np.int)
        trajectory['low_speed_anomaly'] = (speed < self.min_normal_speed).astype(np.int)
        trajectory['airline_anomaly'] = np.ones((len(objective.trajectory,)), dtype=np.int)

        return trajectory

    def generate(self, num, save_in=None):
        trajectories = []
        anomaly_types = random.choices(
            ['normal_safe', 'normal_alert', 'anomaly_straight',
             'anomaly_circle', 'anomaly_random'],
            [self.prob_normal_safe, self.prob_normal_alert, self.prob_anomaly_straight,
             self.prob_anomaly_circle, self.prob_anomaly_random],
            k=num)

        for n, anomaly_type in zip(range(num), anomaly_types):
            if anomaly_type is 'normal_safe':
                trajectory = self.generate_normal_safe_trajectory()
            elif anomaly_type is 'normal_alert':
                trajectory = self.generate_normal_alert_trajectory()
            elif anomaly_type is 'anomaly_straight':
                trajectory = self.generate_anomaly_straight_trajectory()
            elif anomaly_type is 'anomaly_circle':
                r = np.random.uniform(0.5 * self.max_radius, self.max_radius)
                trajectory = self.generate_circular_trajectory(
                    np.random.uniform(self.max_radius - r, self.max_radius + r),
                    np.random.uniform(0, 2 * np.pi),
                    r,
                    np.random.uniform(self.min_height, self.max_height),
                    np.random.uniform(self.min_speed, self.max_speed))
            elif anomaly_type is 'anomaly_random':
                trajectory = self.generate_random_trajectory(
                    (self.min_speed + self.max_speed) / 2,
                    self.alert_height,
                    0, 0.01, 0.01)
            else:
                raise ValueError('Invalid anomaly_type.')
            if save_in is not None and trajectory is not None:
                trajectory.to_csv(os.path.join(save_in, '{}.csv'.format(n)), index=False)
            trajectories.append(trajectory)

        return trajectories

    def plot(self, obj):
        self._airspace.plot(plt, 'r,', color='grey')
        environment.Airspace(self.x_loc, self.y_loc, self.alert_radius).plot(obj, 'r,', color='grey')
        environment.Airspace(self.x_loc, self.y_loc, self.dangerous_radius).plot(obj, 'r,', color='grey')
        for airline in self._normal_safe_airlines + self._normal_alert_airlines:
            airline.plot(obj, 'r:', color='grey')

    def dump(self, file):
        pickle.dump(self, file)

    @classmethod
    def load_from(cls, file):
        return pickle.load(file)


def display(generator: TrajectoryGenerator, trajectories):
    plt.rcParams['figure.figsize'] = [7.2, 7.2]

    for frame in range(10000):
        plt.cla()
        plt.xlim([
            generator.airspace.x_loc - 1.2 * generator.airspace.radius,
            generator.airspace.x_loc + 1.2 * generator.airspace.radius])
        plt.ylim([
            generator.airspace.y_loc - 1.2 * generator.airspace.radius,
            generator.airspace.y_loc + 1.2 * generator.airspace.radius])
        generator.plot(plt)
        for t in trajectories:
            if t is not None:
                plt.plot(t['x'][frame: frame + 1], t['y'][frame: frame + 1], 'o')

        plt.pause(0.01)

    plt.ioff()
    plt.show()
