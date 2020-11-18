import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from mpl_toolkits import mplot3d

class MovingObject:
    # we use hours as time unit and km as distance unit
    deltaT = 1/60 #deltaT of one minute
    initial_position = np.array([10,1,1]) # a_0 = [10,1,1] column vector
    v = 20 
    print(initial_position[0])
    print(initial_position[1])

    time = np.arange(0,initial_position[0]/v,deltaT)
    
    position = np.array([initial_position[0]+ v*time,
                         initial_position[1]* np.sin((4*np.pi*v/initial_position[0]) * time),
                         np.sin((np.pi*v/initial_position[0]) * time)
                         ])
    velocity = np.array([v*time, (np.cos((4*np.pi*v/initial_position[0])*time)*(4*np.pi*v/initial_position[0])),
                             (np.cos((np.pi*v/initial_position[0])*time)*(np.pi*v/initial_position[0]))])
    acc = -q* np.array([time,
                    -1*(np.sin((4*np.pi*v/initial_position[0])*time)*(4*np.pi*v/initial_position[0])**2),
                        -1*(np.sin((np.pi*v/initial_position[0])*time)*(np.pi*v/initial_position[0])**2)])
    

    
moving_object = MovingObject()
position = moving_object.position
vel = moving_object.velocity
acc = moving_object.acc

print(position)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot3D(position[0], position[1], position[2], 'gray')

ax.scatter3D(position[0],position[1], position[2], color='red',label="Ground truth position");

plt.title("3D Plane", fontsize=19)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#plt.tick_params(axis='both', which='major', labelsize=9)
plt.legend(loc='lower right')

fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot(vel[1], vel[2], zs=20, zdir='z',color = 'red' , label='velocity in (x, y)')
plt.title("Velocity")
ax2.set_xlabel('y')
ax2.set_ylabel('z')
ax2.set_zlabel('x')
# Data for a three-dimensional line



fig3 = plt.figure()
ax3 = plt.axes(projection='3d')
ax3.plot(acc[1], acc[2], zs=0, zdir='z', label='acceleration in (x, y)')
plt.title("Acceleration")
ax3.set_xlabel('y')
ax3.set_ylabel('z')
ax3.set_zlabel('x')



plt.show()
