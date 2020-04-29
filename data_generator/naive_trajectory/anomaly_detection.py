import collections
import numpy as np


from matplotlib import pyplot as plt


class GCSPlaneLocation(object):
    def __init__(self, longitude, latitude):
        self.longitude = np.array(longitude)
        self.latitude = np.array(latitude)

    @property
    def shape(self):
        if list(np.shape(self.longitude)) == list(np.shape(self.latitude)):
            return np.shape(self.longitude)
        raise ValueError('Unmatched shape.')


class GCSLocation(GCSPlaneLocation):
    def __init__(self, longitude, latitude, altitude):
        super(GCSLocation, self).__init__(longitude, latitude)
        self.altitude = np.array(altitude)


class Airspace(collections.namedtuple('Airspace', ('centre', 'radius'))):
    def generate_airline(self, distance, angle, height, reverse=False):
        phy = np.arccos(distance / self.radius)
        location_1 = GCSLocation(
            self.centre.longitude + distance * np.cos(angle - phy),
            self.centre.latitude + distance * np.sin(angle - phy),
            height)
        location_2 = GCSLocation(
            self.centre.longitude + distance * np.cos(angle + phy),
            self.centre.latitude + distance * np.sin(angle + phy),
            height)
        if reverse:
            return location_2, location_1
        return location_1, location_2

    def random_airlines(self, num):
        airlines = []
        while len(airlines) <= num:
            r0 = np.random.normal(0.5, 0.3)  #
            if 0 < r0 < 1:
                r0 = r0 * self.radius
                theta0 = np.random.uniform(0., 2 * np.pi)
                phy0 = (np.random.normal(theta0 + 0.5 * np.pi, 0.1))

                a = 1
                b = 2 * r0 * np.cos(theta0 - phy0)
                c = r0 ** 2 - self.radius ** 2
                delta = b ** 2 - 4. * a * c
                t1 = (- b + np.sqrt(delta)) / (2. * a)
                t2 = (- b - np.sqrt(delta)) / (2. * a)

                boundaries = GCSPlaneLocation(
                    [self.centre.longitude + r0 * np.cos(theta0) + t1 * np.cos(phy0),
                     self.centre.longitude + r0 * np.cos(theta0) + t2 * np.cos(phy0)],
                    [self.centre.latitude + r0 * np.sin(theta0) + t1 * np.sin(phy0),
                     self.centre.latitude + r0 * np.sin(theta0) + t2 * np.sin(phy0)])
                airlines.append(boundaries)
        return airlines

    @property
    def space_as_dots(self):
        theta = np.linspace(0, 2 * np.pi, 1000)
        x = self.centre.longitude + self.radius * np.cos(theta)
        y = self.centre.longitude + self.radius * np.sin(theta)
        return x, y

    def generate_normal_trajectory(self, start_location, end_location, velocity, dt=1, start_time=0, with_noise=False):
        x_distance = end_location.longitude - start_location.longitude
        y_distance = end_location.latitude - start_location.latitude
        z_distance = end_location.altitude - end_location.altitude
        distance = np.sqrt(np.square(x_distance) + np.square(y_distance) + np.square(z_distance))

        dx = x_distance / distance
        dy = y_distance / distance
        dz = z_distance / distance
        time = distance / velocity

        t = np.arange(start_time, start_time + time, dt)
        displacement = velocity * t
        longitude = start_location.longitude + displacement * dx
        latitude = start_location.latitude + displacement * dy
        altitude = start_location.altitude + displacement * dz

        if with_noise:
            r = np.random.normal(0., 0.001 * self.radius, size=np.shape(t))
            theta = np.random.normal(np.arctan(y_distance / x_distance) + 0.5 * np.pi, 0.01, size=np.shape(t))
            longitude = longitude + r * np.cos(theta)
            latitude = latitude + r * np.sin(theta)
            altitude = altitude

        return t, longitude, latitude, altitude

    def generate_anomaly_trajectory__central(self, velocity, dt=1, star_time=0, with_noise=False):
        theta = np.random.normal(0, 2 * np.pi)
        start_location = GCSPlaneLocation(
            self.centre.longitude + self.radius * np.cos(theta),
            self.centre.longitude + self.radius * np.sin(theta))
        end_location = self.centre
        return self.generate_normal_trajectory(start_location, end_location, velocity, dt, star_time, with_noise)

    def generate_anomaly_trajectory__too_fast(self, start_location, end_location, dt=1, star_time=0, with_noise=False):
        return self.generate_normal_trajectory(start_location, end_location, dt, star_time, with_noise)

    def generate_anomaly_trajectory__invalid_height(self):
        pass

    def generate_anomaly_trajectory__curve(self):
        pass

    def generate_trajectories(self,
                              airlines,
                              valid_height=(9, 11),
                              valid_velocity=(0.220, 0.330),
                              anomaly_ratio=0.01):
        pass


def display():
    airspace = Airspace(centre=GCSLocation(10., 10., 0.), radius=10.)
    airlines = airspace.random_airlines(20)

    ts, longitudes, latitudes, altitudes = [], [], [], []
    for a in airlines:
        start_location = GCSLocation(a.longitude[0], a.latitude[0], 100)
        end_location = GCSLocation(a.longitude[1], a.latitude[1], 100)
        if np.random.normal() > 0.5:
            start_location, end_location = end_location, start_location
        t, longitude, latitude, altitude = airspace.generate_normal_trajectory(
            start_location,
            end_location,
            0.1, with_noise=True)
        ts.append(t)
        longitudes.append(longitude)
        latitudes.append(latitude)
        altitudes.append(altitude)

    plt.rcParams['figure.figsize'] = [6, 6]
    for i in range(1000000):
        plt.cla()
        plt.plot(*airspace.space_as_dots, 'r,')

        for j in range(len(ts)):
            plt.plot(longitudes[j], latitudes[j], 'r,')

            times = len(ts[j])

            t = max(0, i - 20 * j)  # 进入延迟
            plt.plot(
                longitudes[j][max(0, t % times - 10): t % times],
                latitudes[j][max(0, t % times - 10): t % times])
        plt.pause(0.01)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    display()







