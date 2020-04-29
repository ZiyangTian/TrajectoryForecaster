import numpy as np

from data_generator.anomaly_detection import environment


ANOMALY = {}


class AnomalyRegister(object):
    def __init__(self, name):
        self._name = name

    def __call__(self, cls):
        ANOMALY[self._name] = cls
        return cls


class Anomaly(object):
    """

    """
    def __init__(self, airspace: environment.Airspace, level: int):
        self._airspace = airspace
        self._level = level

    def detect(self, location: environment.Location = None, velocity: environment.Velocity = None, **kwargs):
        raise NotImplementedError('Anomaly.detect')


@AnomalyRegister('distance_anomaly')
class DistanceAnomaly(Anomaly):
    """ 速度方向与中心距离低于`distance` """
    def __init__(self, distance, airspace: environment.Airspace, level=1):
        super(DistanceAnomaly, self).__init__(airspace, level)
        self._distance = distance

    def detect(self, location: environment.Location = None, velocity: environment.Velocity = None, **kwargs):
        velocity_direction = environment.StraightLine2D.from_vector(location.x, location.y, velocity.slope_in_plane)
        distance = velocity_direction.distance_to_point(self._airspace.x_loc, self._airspace.y_loc)
        cond = np.less(distance, self._distance)
        return np.where(cond, np.ones_like(cond, dtype=np.float), np.zeros_like(cond, dtype=np.float))


@AnomalyRegister('height_anomaly')
class HeightAnomaly(Anomaly):
    """ 距离在lower_distance和upper_distance之间且高度低于height """
    def __init__(self, lower_distance, upper_distance, height, airspace: environment.Airspace, level=2):
        super(HeightAnomaly, self).__init__(airspace, level)
        self._lower_distance = lower_distance
        self._upper_distance = upper_distance
        self._height = height

    def detect(self, location=None, velocity=None, **kwargs):
        pass


@AnomalyRegister('high_speed_anomaly')
class HighSpeedAnomaly(Anomaly):
    def __init__(self, upper_speed, airspace: environment.Airspace, level=3):
        super(HighSpeedAnomaly, self).__init__(airspace, level)
        self._upper_speed = upper_speed

    def detect(self, location=None, velocity=None, **kwargs):
        pass


@AnomalyRegister('low_speed_anomaly')
class LowSpeedAnomaly(Anomaly):
    def __init__(self, lower_speed, airspace: environment.Airspace, level=4):
        super(LowSpeedAnomaly, self).__init__(airspace, level)
        self._lower_speed = lower_speed

    def detect(self, location=None, velocity=None, **kwargs):
        pass


@AnomalyRegister('airline_anomaly')
class AirlineAnomaly(Anomaly):
    def __init__(self, airspace: environment.Airspace, level=5):
        super(AirlineAnomaly, self).__init__(airspace, level)

    def detect(self, location=None, velocity=None, **kwargs):
        pass
