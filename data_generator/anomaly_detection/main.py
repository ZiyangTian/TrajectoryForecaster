from data_generator.anomaly_detection import generator


def main():
    gen = generator.TrajectoryGenerator(
        dt=10,
        num_normal_safe_airlines=8,
        num_normal_alert_airlines=2,
        prob_normal_safe=0.3,
        prob_normal_alert=0.1,
        prob_anomaly_straight=0.2,
        prob_anomaly_circle=0.2,
        prob_anomaly_random=0.2,
        prob_distance_anomaly=0.3,
        prob_height_anomaly=0.3,
        prob_high_speed_anomaly=0.6,
        prob_low_speed_anomaly=0.1,
        prob_airline_anomaly=0.6)
    trajectories = gen.generate(30)

    for i, trajectory in enumerate(trajectories):
        trajectory.to_csv('/Users/Tianziyang/projects/AnomalyDetection/data/raw/{}.csv'.format(i), index=False)

    generator.display(gen, trajectories)
    # circle1 = generator.generate_circular_trajectory(160, np.pi / 6, 200, 7, 0.25)
    # circle2 = generator.generate_circular_trajectory(170, np.pi, 210, 7, 0.2)
    # circle3 = generator.generate_circular_trajectory(180, np.pi / 2, 205, 7, 0.3)
    # other_test = generator.generate_random_trajectory(0.25, 0.8, 0, 0.01, 0.01)
    # other_test.to_csv('other_test.csv', index=False)


if __name__ == '__main__':
    main()
