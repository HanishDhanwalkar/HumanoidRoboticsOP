from ikine3dof import RoboticArmEnv, rad_to_deg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from math import pi, sqrt

npooints = 20
theta1 = np.linspace(0, 2*pi, npooints)

R = 6

traj = R*np.cos(theta1) + R * np.sin(theta1)

actual = np.array([0, 0], dtype=np.float64)
sol = []
desired =[]
for i,the in enumerate(theta1):
    desx, desy = 0+R*np.cos(the), 0+R*np.sin(the)
    desired_position = np.array([desx, desy ], dtype=np.float32)
    desired.append(desired_position)
    intial_guess = actual

    env = RoboticArmEnv(intial_guess, desired_position)
    obs = env.reset(intial_guess, desired_position)
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        # print("Action:", action, "Observation:", obs, "Reward:", reward)
    # print("Total Reward:", total_reward)
    # for i, j in zip(['current_x', 'current_y', 'desired_x', 'desired_y', 'joint1_angle', 'joint2_angle', 'joint3_angle'], obs):
        
    #     if i == 'joint1_angle':
    #         print("joint1_angle: ", rad_to_deg(j))
    #     elif i == 'joint2_angle':
    #         print("joint2_angle: ", rad_to_deg(j))
    #     elif i == 'joint3_angle':
    #         print("joint3_angle: ", rad_to_deg(j))
    #     else:
    #         print(i, ": ", j)
    print(f'{i}th point')
    print(obs[-3:])
    actual = obs[-3:]
    sol.append(actual)

    # joint_angles = obs[-3:]

def animate(i):
    joint_angles = sol[i]
    plt.cla()
    for point in desired:
        plt.plot(point[0], point[1], 'rx')

    x_coords, y_coords = [], []
    x_coords.append(0)
    y_coords.append(0)
    for j, angle in enumerate(joint_angles):
        x_coords.append(x_coords[-1] + env.arm_length[j] * np.cos(angle))
        y_coords.append(y_coords[-1] + env.arm_length[j] * np.sin(angle))

    plt.plot([0, env.arm_length[0]*np.cos(joint_angles[0])],
                [0, env.arm_length[0]*np.sin(joint_angles[0])], 'r-')
    plt.plot([env.arm_length[0]*np.cos(joint_angles[0])],
                [env.arm_length[0]*np.sin(joint_angles[0])], 'ko')
    plt.plot([env.arm_length[0]*np.cos(joint_angles[0]),
                env.arm_length[0]*np.cos(joint_angles[0]) + env.arm_length[1]*np.cos(joint_angles[0] + joint_angles[1])],
                [env.arm_length[0]*np.sin(joint_angles[0]),
                env.arm_length[0]*np.sin(joint_angles[0]) + env.arm_length[1]*np.sin(joint_angles[0] + joint_angles[1])], 'g-')
    plt.plot([env.arm_length[0]*np.cos(joint_angles[0]) + env.arm_length[1]*np.cos(joint_angles[0] + joint_angles[1])],
                [env.arm_length[0]*np.sin(joint_angles[0]) + env.arm_length[1]*np.sin(joint_angles[0] + joint_angles[1])], 'ko')
    plt.plot([env.arm_length[0]*np.cos(joint_angles[0]) + env.arm_length[1]*np.cos(joint_angles[0] + joint_angles[1]),
                env.arm_length[0]*np.cos(joint_angles[0]) + env.arm_length[1]*np.cos(joint_angles[0] + joint_angles[1]) + env.arm_length[2]*np.cos(joint_angles[0] + joint_angles[1] + joint_angles[2])],
                [env.arm_length[0]*np.sin(joint_angles[0]) + env.arm_length[1]*np.sin(joint_angles[0] + joint_angles[1]),
                env.arm_length[0]*np.sin(joint_angles[0]) + env.arm_length[1]*np.sin(joint_angles[0] + joint_angles[1]) + env.arm_length[2]*np.sin(joint_angles[0] + joint_angles[1] + joint_angles[2])], 'b-')
    plt.plot([env.arm_length[0]*np.cos(joint_angles[0]) + env.arm_length[1]*np.cos(joint_angles[0] + joint_angles[1]) + env.arm_length[2]*np.cos(joint_angles[0] + joint_angles[1] + joint_angles[2])],
                [env.arm_length[0]*np.sin(joint_angles[0]) + env.arm_length[1]*np.sin(joint_angles[0] + joint_angles[1]) + env.arm_length[2]*np.sin(joint_angles[0] + joint_angles[1] + joint_angles[2])], 'ko')
    
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.gca().set_aspect('equal', adjustable='box')

# Create the animation object
fig, ax = plt.subplots() 

ani = FuncAnimation(fig, animate, frames=20, interval=0.1)  


plt.show()
