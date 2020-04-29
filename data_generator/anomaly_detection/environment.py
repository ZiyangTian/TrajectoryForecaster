import abc
import collections
import functools
import typing
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


class Coordinate3D(collections.namedtuple('Coordinate3D', ('x', 'y', 'z'))):
    def __new__(cls, *args, **kwargs):
        np_float_scalar = functools.partial(np.array, dtype=np.float)
        if len(args) == 1 and not kwargs:
            arg = args[0]
            if isinstance(arg, cls):
                return arg
            if isinstance(arg, (typing.Sized, typing.Sequence)):
                return super(Coordinate3D, cls).__new__(cls, *map(np_float_scalar, arg))

        args = tuple(map(np_float_scalar, args))
        for k, v in kwargs.items():
            kwargs[k] = np_float_scalar(v)
        self = super(Coordinate3D, cls).__new__(cls, *args, **kwargs)
        if not self.x.shape == self.y.shape == self.z.shape:
            raise ValueError('`x`, `y` and `z` must be of the same shape, got {}, {}, and {}.'.format(
                self.x.shape, self.y.shape, self.z.shape))
        return self

    def __add__(self, other):
        other = type(self)(other)
        return Coordinate3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z)

    def __neg__(self):
        return type(self)(-self.x, -self.y, -self.z)

    def __sub__(self, other):
        return self + type(self)(other).__neg__()

    def __mul__(self, other):
        return type(self)(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def dot(self, other):
        other = type(self)(other)
        return self.x * other.x + self.y * other.y + self.z * other.z

    @property
    def shape(self):
        return self.x.shape

    @property
    def modulus(self):
        return np.sqrt(np.sum(np.square(self.numpy), axis=-1))

    @property
    def slope_in_plane(self):
        return np.where(
            np.logical_and(np.equal(self.x, 0), np.equal(self.y, 0)),
            np.zeros(self.shape),
            np.true_divide(self.y, self.x))

    @property
    def numpy(self):
        if self.shape is ():
            return np.array([self.x, self.y, self.z])
        return np.concatenate([self.x, self.y, self.z], axis=-1)

    def should_be_scalar(self, name):
        if self.shape is not ():
            raise AssertionError('`{}` must be a scalar.'.format(name))
        return self


class Location(Coordinate3D):
    def compute_distance(self, other):
        return (self - other).modulus


class Velocity(Coordinate3D):
    pass

    @property
    def theta(self):
        modulus = self.modulus
        return np.where(
            np.equal(modulus, 0),
            np.zeros_like(modulus, dtype=np.float),
            np.arcsin(np.sqrt(np.square(self.x) + np.square(self.y)) / self.modulus))

    @property
    def phy(self):
        modulus = self.modulus


    # def xy_angle(self):
    #     """ [-pi, pi] """
    #     if self.x == self.y == 0:
    #         return 0
    #     if self.x == 0:
    #         if self.y > 0:
    #             return 0.5 * np.pi
    #         if self.y < 0:
    #             return -0.5 * np.pi
    #     if self.x > 0:
    #         return np.arctan(self.y / self.x)
    #     if self.y >= 0:
    #         return np.pi + np.arctan(self.y / self.x)


class StraightLine2D(collections.namedtuple('StraightLine2D', ('A', 'B', 'C'))):
    def distance_to_point(self, x, y):
        return np.abs(self.A * x + self.B * y + self.C) / np.sqrt(np.square(self.A) + np.square(self.B))

    @classmethod
    def from_vector(cls, x, y, slope):
        cond = np.logical_or(np.equal(slope, np.inf), np.equal(slope, -np.inf))
        a = np.where(cond, np.ones_like(cond, dtype=np.float), slope)
        b = np.where(cond, np.zeros_like(cond, dtype=np.float), - np.ones_like(cond, dtype=np.float))
        c = np.where(cond, -x, y - slope * x)
        return cls(a, b, c)


class StraightAirline(collections.namedtuple('Airline', ('starting_location', 'ending_location'))):
    @property
    def slope(self):
        dx = self.ending_location.x - self.starting_location.x
        dy = self.ending_location.y - self.starting_location.y
        return np.true_divide(dy, dx)

    def plot(self, obj, *args, **kwargs):
        obj.plot(
            [self.starting_location.x, self.ending_location.x],
            [self.starting_location.y, self.ending_location.y],
            *args, **kwargs)


class Airspace(collections.namedtuple('Airspace', ('x_loc', 'y_loc', 'radius'))):
    def generate_straight_airline(self, distance, angle, height, clockwise=False):
        phy = np.arccos(distance / self.radius)
        location_1 = Location(
            self.x_loc + self.radius * np.cos(angle - phy),
            self.y_loc + self.radius * np.sin(angle - phy),
            height)
        location_2 = Location(
            self.x_loc + self.radius * np.cos(angle + phy),
            self.y_loc + self.radius * np.sin(angle + phy),
            height)
        if clockwise:
            return StraightAirline(location_2, location_1)
        return StraightAirline(location_1, location_2)

    def plot(self, obj, *args, **kwargs):
        theta = np.linspace(0, 2 * np.pi, 1000)
        x = self.x_loc + self.radius * np.cos(theta)
        y = self.y_loc + self.radius * np.sin(theta)
        obj.plot(x, y, *args, **kwargs)


class Objective(object):
    _FEATURES = ['t', 'x', 'y', 'z']

    def __init__(self, initial_location: Location):
        self._t = 0.
        self._location = initial_location.should_be_scalar('initial_location')
        self._trajectory = pd.DataFrame(dict(zip(self._FEATURES, (
            [0.], [initial_location.x], [initial_location.y], [initial_location.z]))), dtype=np.float)

    def to_csv(self, *args, **kwargs):
        return self._trajectory.to_csv(*args, **kwargs)

    def plot(self, obj, *args, **kwargs):
        obj.plot(self._trajectory['x'], self._trajectory['y'], *args, **kwargs)

    @property
    def t(self):
        return self._t

    @property
    def location(self):
        return self._location

    @property
    def trajectory(self):
        return self._trajectory


class UniformLinearObjective(Objective):
    def __init__(self, initial_location: Location, velocity: Velocity):
        super(UniformLinearObjective, self).__init__(initial_location)
        self._velocity = velocity.should_be_scalar('velocity')

    def move(self, dt):
        self._t += dt
        self._location += self._velocity * dt
        self._trajectory.append({
            't': self._t,
            'x': self._location.x,
            'y': self._location.y,
            'z': self._location.z})

    def move_n(self, time, dt):
        t = np.arange(dt, time + dt, dt)
        displacement = self._velocity * t
        t = self._t + t
        displacement = self._location + displacement

        self._t += time
        self._location = self._velocity * time
        self._trajectory = pd.concat([
            self._trajectory,
            pd.DataFrame({
                't': t,
                'x': displacement.x,
                'y': displacement.y,
                'z': displacement.z})])

    def get_noised(self, stddev):
        trajectory = self._trajectory.copy()
        r = np.random.normal(0., stddev, size=(len(trajectory),))
        theta = np.random.normal(np.arctan(self._velocity.slope_in_plane) + 0.5 * np.pi, 0.01, size=(len(trajectory),))
        trajectory['x'] += r * np.cos(theta)
        trajectory['y'] += r * np.sin(theta)
        return trajectory


class UniformCircumferenceObjective(Objective):
    def __init__(self, initial_location: Location, centre: Location, speed):
        super(UniformCircumferenceObjective, self).__init__(initial_location)
        self._centre = centre
        self._radius = centre.compute_distance(initial_location)
        self._speed = speed

        self._theta = None

    def _move(self, dt):
        pass


class RandomObjective(Objective):
    def __init__(self, initial_location: Location, initial_velocity: Velocity,
                 acceleration_min, acceleration_max, height_stddev, speed_min=0, speed_max=None):
        super(RandomObjective, self).__init__(initial_location)
        self._velocity = initial_velocity.should_be_scalar('velocity')
        self._trajectory['vx'] = [self._velocity.x]
        self._trajectory['vy'] = [self._velocity.y]
        self._trajectory['vz'] = [self._velocity.z]
        self._acceleration_min = acceleration_min
        self._acceleration_max = acceleration_max
        self._height_stddev = height_stddev
        self._speed_min = speed_min
        self._speed_max = speed_max

    def move(self, dt):
        self._t += dt
        dv_rou = np.random.uniform(self._acceleration_min, self._acceleration_max) * dt
        dv_rou = np.clip(dv_rou, self._speed_min, self._speed_max)
        dv_theta = np.random.uniform(0, 2 * np.pi)
        self._velocity += (
            dv_rou * np.cos(dv_theta),
            dv_rou * np.sin(dv_theta),
            np.random.normal(scale=self._height_stddev))
        self._location += self._velocity * dt
        self._trajectory = pd.concat([self._trajectory, pd.DataFrame({
            't': [self._t],
            'x': [self._location.x],
            'y': [self._location.y],
            'z': [self._location.z],
            'vx': [self._velocity.x],
            'vy': [self._velocity.y],
            'vz': [self._velocity.z]})])

    def get_noised(self, stddev):
        trajectory = self._trajectory.copy()
        r = np.random.normal(0., stddev, size=(len(trajectory),))
        v = Velocity(self._trajectory['vx'], self._trajectory['vy'], self._trajectory['vz'])
        theta = np.random.normal(np.arctan(v.slope_in_plane) + 0.5 * np.pi, 0.01)
        trajectory['x'] += r * np.cos(theta)
        trajectory['y'] += r * np.sin(theta)

        del trajectory['vx']
        del trajectory['vy']
        del trajectory['vz']
        return trajectory


