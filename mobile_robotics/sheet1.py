#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 14:38:27 2021

@author: dhruvrajgupta
"""

import math
import numpy as np
import matplotlib.pyplot as plt

pi = math.pi
scan = np.loadtxt('laserscan.dat')
angle = np.linspace(start=-pi/2, stop=pi/2, 
                    num=np.shape(scan)[0], endpoint=True)

# Plotting the laser end points with reference of laser range finder
x_l = scan * np.cos(angle)
y_l = scan * np.sin(angle)

"""
plt.plot(x_l,y_l, '.k', markersize=3)
# Set the same scale on both axes
plt.gca().set_aspect('equal')
plt.savefig('laserscan1.pdf')
plt.show()
"""


# transformation matrices
T_global_robot = np.array([
    [ np.cos(pi/4), -np.sin(pi/4), 1.0 ],
    [ np.sin(pi/4), np.cos(pi/4), 0.5 ],
    [ 0, 0, 1 ]    
])

T_robot_laser = np.array([
    [ np.cos(pi), -np.sin(pi), 0.2 ],
    [ np.sin(pi), np.cos(pi), 0.0 ],
    [0, 0, 1 ]
])

# laser frame wrt to global
T_global_laser = np.matmul(T_global_robot, T_robot_laser)

# laser ponts wrt to global
w = np.ones(len(x_l))
scan_laser = np.array([x_l, y_l, w])
scan_global = np.matmul(T_global_laser, scan_laser)


# Plotting laser points, robot pose and laser pose
plt.figure()
plt.plot(scan_global[0,:], scan_global[1,:], '.k', markersize=3)

plt.plot(T_global_robot[0,2], T_global_robot[1,2], '+r')

plt.plot(T_global_laser[0,2], T_global_laser[1,2], '+b')


plt.gca().set_aspect('equal')
plt.savefig('laser_scan.pdf')
plt.show()