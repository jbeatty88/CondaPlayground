from matplotlib import pyplot as plt
import numpy as np


plt.title('My graph')
plt.ylabel('Y Axis')
plt.xlabel('X axis')

# EZ CHALLENGE
# t = np.arange(0.0, 2.0, 0.01)
# s = np.sin(2 * np.pi * t)
#
# plt.plot(t, s, "c", color="blue")
# plt.xlim(0, 5)
# plt.ylim(-1.5, 1.5)
#
# plt.grid()
# plt.show()



#INTER CHALLENGE
N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)
menStd = (2, 3, 4, 1, 2)
womenStd = (3, 5, 2, 3, 3)
ind = np.arange(N)    # the x locations for the groups

plt.title("Scores by group and gender")
plt.bar(ind, menMeans, color="blue", yerr=N, alpha=0.4)
plt.bar(ind, womenMeans, color="red", yerr=N, alpha=0.4)
# plt.errorbar(menStd, menMeans)
plt.ylim(0, 80)
plt.show()

#
# divisions = [1,2,3,4]
# divisions_averages = [2, 1.5, 10, 16]
# plt.bar(divisions, divisions_averages, color="blue", alpha=0.4)
#
# plt.title("GRAPH")
#
# plt.xlim(0,5)
# plt.ylim(0,20)
#
# plt.show()
#
#
# height=np.array([1, 5, 65, 23, 15, 89, 46, 25, 35, 78, 12, 45, 67, 45, 23, 34, 67, 99])
# weight=np.array([6, 53, 2, 67, 23, 45, 66, 77,58, 99, 11, 33, 45, 67, 34, 45, 86, 61])
#
# # plt.xlim(140, 200)
# # plt.ylim(60,100)
# # plt.scatter(height, weight)
# # plt.show()
#
# from mpl_toolkits import mplot3d
#
# ax = plt.axes(projection='3d')
# ax.scatter3D(height, weight)
# plt.show()







