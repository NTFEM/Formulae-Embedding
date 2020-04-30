import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# size = 20
# x = np.arange(size)
a = np.array([0.2457,0.4831,0.6334,0.7579,
              0.5714,0.8056,0.4503,0.6204,
              0.6358,0.6061,0.9333,0.7633,
              0.1684,0.4793,0.6524,0.3062,
              0.7724,0.2035,0.3305,0.4174])
b = np.array([0.3333,0.5505,0.2500,0.6944,
              0.7500,0.6582,0.3750,0.5306,
              1.0,1.0,0.9136,0.9256,
              0.1000,0.1667,1.0,0.2569,
              1.0,0.4400,1.0,0.3100])

size = 20
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
# a = np.array([0.4831,0.7579,
#               0.8056,0.6204,
#               0.6061,0.7633,
#               0.4793,0.3062,
#               0.2035,0.4174])
#
# b = np.array([0.5505,0.6944,
#               0.6582,0.5306,
#               1.0,0.9256,
#               0.1667,0.2569,
#               0.4400,0.3100])

# a = np.random.random(size)
# b = np.random.random(size)
# c = np.random.random(size)

total_width, n = 0.6, 2
width = total_width / n
x = x - (total_width - width) / 2

plt.figure(figsize=(12,5))
plt.bar(x, a ,width=width, color = "cornflowerblue",label='partial relevance')
plt.bar(x + width, b, width=width, color = "sandybrown" ,label='full relevance')
plt.xlabel('Queries')
plt.ylabel('bpref score')
# x_ticks = np.linspace(0,20,11)
x_ticks = np.arange(1,21,1)
y_ticks = np.arange(0,1.2,0.2)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
# plt.bar(x + 2 * width, c, width=width, label='c')
plt.legend()

plt.savefig("static.eps", format="eps")
# plt.show()