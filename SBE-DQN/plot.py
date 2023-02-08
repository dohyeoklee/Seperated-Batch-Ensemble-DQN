import numpy as np, scipy.stats as st
import sys
import matplotlib.pyplot as plt
import torch
import math

num_data = 10
epi_len = 500
moving_average = 10
model = ["dqn", "ensemble_dqn", "SBE_dqn"]
env_name = 'CartPole-v1_10'
#env_name = 'MountainCar-v0'
#env_name = 'Acrobot-v1'
mean_data = []
std_data = []

for md in model:
    data = []
    for i in range(1, num_data + 1):
        f = open("./{}/{}_{}.txt".format(env_name, md, i), 'r')
        line = f.readline()
        line_list = line.split()
        line_list_float = list(map(float, line_list))
        mv_avg = np.convolve(line_list_float, np.ones(moving_average), 'valid') / moving_average
        data.append(mv_avg)

    np_data = np.array(data)
    mean = np.mean(np_data, axis = 0)
    std = np.std(np_data, axis = 0)

    mean_data.append(mean)
    std_data.append(std)

x_axis = list(range(1, len(mean_data[0]) + 1))

plt.plot(x_axis, mean_data[0], c = 'green', label = 'dqn')
plt.plot(x_axis, mean_data[1], c = 'blue', label = 'ensemble_dqn')
plt.plot(x_axis, mean_data[2], c = 'red', label = 'ours')

plt.fill_between(x_axis, mean_data[0]- std_data[0], mean_data[0] + std_data[0], alpha = 0.3, edgecolor = 'green', facecolor = 'green', linewidth=0, antialiased = True)
plt.fill_between(x_axis, mean_data[1]- std_data[1], mean_data[1] + std_data[1], alpha = 0.3, edgecolor = 'blue', facecolor = 'blue', linewidth=0, antialiased = True)
plt.fill_between(x_axis, mean_data[2]- std_data[2], mean_data[2] + std_data[2], alpha = 0.3, edgecolor = 'red', facecolor = 'red', linewidth=0, antialiased = True)

plt.title('CartPole-v1')
plt.xlabel('num_episode')
plt.ylabel('mean_test_return')
plt.legend()
plt.savefig("./plot/{}num_seed_{}.png".format(env_name, num_data))
plt.show()