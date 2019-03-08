import argparse
import numpy as np
import matplotlib.pyplot as plt

__DEBUG__ = 0


class EKF():

    def __init__(self, front_wheel_radius, r_dist, ticks_per_rev, meters_per_tick, n_states=7, n_observations=1):
        self.front_wheel_radius = front_wheel_radius
        self.r_dist = r_dist
        self.ticks_per_rev = ticks_per_rev
        self.meters_per_tick = meters_per_tick
        self.dt = 0

        self.N = n_states  # Number of states
        self.M = n_observations  # Number of observations
        self.X = np.zeros((self.N, 1), dtype=float)
        self.F = np.zeros((self.N, self.N), dtype=float)
        # self.P = np.zeros((self.N, self.N), dtype=float)
        self.Q = 0.0001 * np.identity(self.N, dtype=float)
        self.R = 0.00001 * np.identity(self.M, dtype=float)
        self.I = np.identity(self.N, dtype=float)
        self.Z = np.zeros((self.M, 1), dtype=float)

        self.prev_state = np.array([[0, 0, 0, 0, 0, 0, 0]], dtype=float)
        self.prev_time = 0
        self.prev_ticks = 0

    # This function calculates translational velocity (v) and angular velocity (omega)
    # from the time difference and encoder_ticks difference.
    def calc_input(self, t, ticks, steering_angle):
        # Get Time difference
        self.dt = t - self.prev_time

        # Get ticks difference
        d_enc = ticks - self.prev_ticks

        # Get distance travelled by wheel in dt time
        d_distance = d_enc * self.meters_per_tick

        # Find wheel velocity in dt time
        v_enc = d_distance / float(self.dt)

        # The forward (translational) velocity of robot (in Robot's X axis)
        v = v_enc * np.cos(steering_angle)
        # The angular velocity of the robot (About Robot's Z axis)
        omega = (v_enc * np.sin(steering_angle)) / self.r_dist

        # Update the time and ticks counters
        self.prev_time = t
        self.prev_ticks = ticks

        # Return input vector comprised of v and omega
        u = np.array([[v, omega]]).T
        return u

    def observation(self, xTrue, xd, u):
        # Predict state based on input
        xTrue = self.motion_model(xTrue, u)

        # Add Noise to IMU (observation)
        z = xTrue[6, 0] # + np.random.randn() * np.deg2rad(30) ** 2])

        ud1 = u[0, 0] # + np.random.randn() * 1
        ud2 = u[1, 0] # + np.random.randn() * np.deg2rad(30) ** 2
        ud = np.array([[ud1, ud2]]).T

        xd = self.motion_model(xd, ud)

        return xTrue, z, xd, ud

    def motion_model(self, x, u):
        # x = [x, y, theta, x_dot, theta_dot, v_enc, alpha]
        F = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=float)

        B = np.array([[np.cos(x[2, 0]) * self.dt, 0],
                      [np.sin(x[2, 0]) * self.dt, 0],
                      [0, self.dt],
                      [1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0]], dtype=float)

        x = F.dot(x) + B.dot(u)
        x[2, 0] = np.mod(x[2, 0], 2 * np.pi)
        return x

    def observation_model(self, x):
        # H = np.array([[0, 0, 0, 0, 0, 0, 1],
        # 			  [0, 0, 0, 0, 0, 1, 0]
        # 			  [0, 0, 0, 0, 1, 0, 0]
        # 			  [0, 0, 0, 0, 1, 0, 0]], dtype=float)

        H = np.array([[0, 0, 0, 0, 1.0, 0, 0]], dtype=float)
        z = H.dot(x)

        return z

    def jacobF(self, x, u):

        jF = np.array(
            [[1.0, 0, -self.dt * u[0, 0] * np.sin(x[2, 0]), self.dt * np.cos(x[2, 0]), 0, 0, 0],
             [0, 1.0, -self.dt * u[0, 0] * np.cos(x[2, 0]), self.dt * np.sin(x[2, 0]), 0, 0, 0],
             [0, 0, 1.0, 0, self.dt, 0, 0],
             [0, 0, 0, 0, 0, np.cos(x[6, 0]), 0],
             [0, 0, 0, 0, 0, np.sin(x[6, 0]) / self.r_dist, 0],
             [0, 0, 0, 0, 0, 1.0, 0],
             [0, 0, 0, 0, 0, 0, 1.0]], dtype=float)
        return jF

    def jacobH(self, x):
        # jH = np.array([[0, 0, 0, 0, 0, 0, 1],
        # 			   [0, 0, 0, 0, 0, 1, 0],
        # 			   [0, 0, 0, 0, 0, np.sin(alpha)/self.r_dist, 0],
        # 			   [0, 0, 0, 0, 1, 0, 0]], dtype=float)

        jH = np.array([[0, 0, 0, 0, 1.0, 0, 0]], dtype=float)
        return jH

    # This function runs the entire Kalman filter
    # Args:
    # XEst: Previous state
    # PEst: Previous predicted error
    # u: Current input (in our case, omega from odometer)
    # z: Current observation (in our case, omega from gyro)
    def ekf_estimation(self, xEst, PEst, u, z):
        # Predict
        xPred = self.motion_model(xEst, u)

        jF = self.jacobF(xPred, u)

        PPred = jF.dot(PEst).dot(jF.T) + self.Q

        # Update
        zPred = self.observation_model(xPred)
        jH = self.jacobH(xPred)

        # Error term
        y = z.T - zPred

        S = np.array(jH.dot(PPred).dot(jH.T) + self.R, dtype=float)
        if __DEBUG__:
            print(S.shape)
            print(PPred.dot(jH.T).shape)
            print(jH.shape)

        # If R = 0, S becomes singular
        # if S == 0:
        #     K = (PPred.dot(jH.T)).dot(S)
        # else:
        #     K = (PPred.dot(jH.T)).dot(1/S)#np.linalg.inv(S))
        # Get Kalman gain
        K = (PPred.dot(jH.T)).dot(np.linalg.inv(S))
        # Update state according to Kalman gain
        xEst = xPred + (K).dot(y)

        # Update state error prediction
        PEst = (np.eye(len(xEst)) - (K).dot(jH)).dot(PPred)

        return xEst, PEst


def main():
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
    meters_per_tick = (2.0 * np.pi * front_wheel_radius) / ticks_per_rev

    steering_angle_limit = np.pi / 2.0

    ekf = EKF(front_wheel_radius, r_dist, ticks_per_rev, meters_per_tick)
    xEst = np.zeros((ekf.N, 1))
    PEst = 0.00001 * np.eye(7)

    # History
    xHistory = []
    yHistory = []
    thetaHistory = []

    for counter in range(0, len(time)):
        if counter == 0:
            ekf.prev_time = time[counter]
            ekf.prev_ticks = encoder_ticks[counter]
            # xEst[4, 0] = angular_velocity[counter]
            # xEst[6, 0] = steering_angle[counter]
        else:
            # Update the current steering angle in state
            xEst[6, 0] = steering_angle[counter]

            # Generate input u = [v, omega] vector from current time, current encoder ticks and current steering angle
            u = ekf.calc_input(time[counter], encoder_ticks[counter], steering_angle[counter])

            # xTrue, z, xDR, ud = ekf.observation(xTrue, xDR, u)

            # Run the Kalman Filter
            xEst, PEst = ekf.ekf_estimation(xEst, PEst, u, angular_velocity[counter])

            if __DEBUG__:
                print('Process Covariance Matrix : {} \n'.format(PEst))

            print('X-coordinate  (m)  : {}'.format(xEst[0, 0]))
            print('Y-coordinate  (m)  : {}'.format(xEst[1, 0]))
            print('Heading angle (rad): {}\n'.format(xEst[2, 0]))

            xHistory.append(xEst[0, 0])
            yHistory.append(xEst[1, 0])
            thetaHistory.append(xEst[2, 0])

    if __DEBUG__:
        plt.figure()
        plt.plot(thetaHistory)
        # plt.plot((encoder_ticks[1:len(encoder_ticks)] - encoder_ticks[0:len(encoder_ticks)-1]) * meters_per_tick)
        plt.show()

    # ---------------------- Plotting Trajectories ------------------------------
    # plt.figure()
    # plt.scatter(xHistory, yHistory)
    # plt.title('State Estimate (x,y) Plot for circle_path \n Using Extended Kalman Filter')
    # plt.xlabel('x-coordinate')
    # plt.ylabel('y-coordinate')
    # # plt.show()
    # plt.savefig('output/circle_path_x_y_ekf.png')
    #
    # plt.figure()
    # plt.plot(thetaHistory)
    # plt.title('State Estimate (theta) Plot for circle_path \n Using Extended Kalman Filter')
    # plt.xlabel('Number of data points')
    # plt.ylabel('theta')
    # # plt.show()
    # plt.savefig('output/circle_path_theta_ekf.png')


if __name__ == '__main__':
    main()
