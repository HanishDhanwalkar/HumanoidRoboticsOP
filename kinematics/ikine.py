import numpy as np
from scipy.optimize import minimize

# change these values of desired position and orientation
desired_x = 205
desired_y= 0
desired_z= 16

position_desired = np.array([desired_x, desired_y, desired_z])
orientation_desired = np.array([
        [0,     -1,      0],
        [1,      0,      0],
        [0,      0,      1]
])

# DH Parameters
a =             [0,     16,         60,         129 ]     
alpha =         [0,     -np.pi/2,   np.pi/2,    0   ] 
d =             [0,     16,         0,          0   ] 
theta_initial = [0 ,    0,          0,          0   ] 


def forward_kinematics(theta):
    T = np.eye(4)
    for i in range(len(theta)):
        T_i = np.array([
            [np.cos(theta[i]), -np.sin(theta[i]) * np.cos(alpha[i]), np.sin(theta[i]) * np.sin(alpha[i]), a[i] * np.cos(theta[i])],
            [np.sin(theta[i]), np.cos(theta[i]) * np.cos(alpha[i]), -np.cos(theta[i]) * np.sin(alpha[i]), a[i] * np.sin(theta[i])],
            [0, np.sin(alpha[i]), np.cos(alpha[i]), d[i]],
            [0, 0, 0, 1]
        ])
        T = np.dot(T, T_i)

    position = T[:3, 3]
    orientation = T[:3, :3]

    return position, orientation

# Define the objective func
# --> minimizing the error in pos & ori
def objective(theta):
    position, orientation = forward_kinematics(theta)

    error_position = position_desired - position
    error_orientation = 0.5 * (np.cross(orientation[:, 0], orientation_desired[:, 0]) + np.cross(orientation[:, 1], orientation_desired[:, 1]) + np.cross(orientation[:, 2], orientation_desired[:, 2]))
    error = np.concatenate((error_position, error_orientation))

    norm_error = np.linalg.norm(error)

    return norm_error

# using scipy to minimize the objective func
result = minimize(objective, theta_initial, method='L-BFGS-B', bounds=[(-np.pi, np.pi)] * 4)

theta_optimal = result.x

row_theat_op = theta_optimal.shape[0]

for i in range(row_theat_op):
    if abs(theta_optimal[i]) < 0.001:
        theta_optimal[i] = 0

print("Optimal joint angles:", theta_optimal)
