import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

__DEBUG__ = 1

def estimate(t, ticks, omega, alpha):
    global prev_time, prev_ticks, prev_state

    # Get Time difference
    dt = t - prev_time

    # Find wheel velocity from encoder difference
    d_enc = ticks - prev_ticks
    d_distance = d_enc * ticks_per_meter
    v_enc = d_distance / float(dt)

    # Update the estimator state
    current_state = np.array(prev_state)

    # current_state = [x, y, heading_angle, x_dot, y_dot, angular_velocity]
    current_state[0] += dt * prev_state[3]      # X position of the robot (meters)
    current_state[1] += dt * prev_state[4]      # Y position of the robot (meters)
    current_state[2] += dt * prev_state[5]      # Heading of the robot (rad)
    current_state[2] = np.mod(current_state[2], 2 * np.pi)    # Wrap the heading angle between 0 - 2*pi

    current_state[3] = v_enc * np.cos(alpha) * np.cos(current_state[2])     # X velocity of the robot
    current_state[4] = v_enc * np.cos(alpha) * np.sin(current_state[2])     # Y velocity of the robot
    # current_state[5] = v_enc * np.sin(alpha) / r_dist # Angular velocity of the robot about Z axis (from odometry)
    current_state[5] = omega  # Angular velocity of the robot about Z axis (from gyro)

    prev_state = current_state
    prev_time = t
    prev_ticks = ticks
    return ([current_state[0], current_state[1], current_state[2], current_state[5]])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='State Estimator for BrainCorp by Nishant Shah.')
    parser.add_argument('-d', '--dataset', type=str, default='odom_estimator_dataset/dataset0.csv',
                        help='Dataset.csv file to run the Estimator on.')
    parser.add_argument('-s', '--savedir', type=str, default='output/', help='Directory to save Estimator output.')

    args = parser.parse_args()

    # Load data from the csv file
    mydata = np.genfromtxt(args.dataset, delimiter=',')
    time = mydata[:, 0]
    encoder_ticks = mydata[:, 1]
    angular_velocity = mydata[:, 2]
    steering_angle = mydata[:, 3]

    if __DEBUG__:
        print(time[0], encoder_ticks[0], angular_velocity[0], steering_angle[0])

    # -----------------------------------------------------------------------#
    # Estimator variables

    # Tricycle model parameters
    front_wheel_radius = 0.125  # Front wheel radius
    back_wheel_radius = 0.125  # Rear wheels radius

    d_dist = 0.3673  # Distance b/w rear wheel (width of tricycle)
    r_dist = 0.964  # Distance b/w front and rear wheel (length of tricycle)

    ticks_per_rev = 35136.0  # Encoder resolution
    ticks_per_meter = (2.0 * np.pi * front_wheel_radius) / ticks_per_rev

    steering_angle_limit = np.pi / 2.0

    prev_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    xHistory = []
    yHistory = []
    thetaHistory = []
    omegaHistory = []

    for counter in range(0, len(time)):  # len(time)):
        if counter == 0:
            prev_time = time[counter]
            prev_ticks = encoder_ticks[counter]
            prev_state[5] = angular_velocity[counter]
        else:
            xEst, yEst, thetaEst, omegaEst = estimate(time[counter], encoder_ticks[counter], angular_velocity[counter],
                                            steering_angle[counter])

            print('X-coordinate  (m)  : {}'.format(xEst))
            print('Y-coordinate  (m)  : {}'.format(yEst))
            print('Heading angle (rad): {}\n'.format(thetaEst))

            xHistory.append(xEst)
            yHistory.append(yEst)
            thetaHistory.append(thetaEst)
            omegaHistory.append(omegaEst)

    # ------------------- Plotting Trajectories --------------------------------
    # plt.figure()
    # plt.scatter(xHistory, yHistory)
    # plt.title('State Estimate (x,y) Plot for circle_path \n Using only IMU for angular velocity')
    # plt.xlabel('x-coordinate')
    # plt.ylabel('y-coordinate')
    # # plt.show()
    # plt.savefig('output/circle_path_x_y_imu.png')
    #
    # plt.figure()
    # plt.plot(thetaHistory)
    # plt.title('State Estimate (theta) Plot for circle_path \n Using only IMU for angular velocity')
    # plt.xlabel('Number of data points')
    # plt.ylabel('theta')
    # # plt.show()
    # plt.savefig('output/circle_path_theta_imu.png')