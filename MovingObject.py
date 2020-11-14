
import math
import numpy as np
import matplotlib.pyplot   as plt
import numpy.linalg as la
from mpl_toolkits import mplot3d

class MovingObject:
    # we use hours as time unit and km as distance unit
    deltaT = 1/60 #deltaT of one minute
    initial_position = np.array([10,1,1]) # a_0 = [10,1,1] column vector
    v = 20 
    time = np.arange(0,initial_position[0]/v,deltaT)
    
    position = np.array([initial_position[0]+ v*time,
                         initial_position[1]* np.sin((4*np.pi*v/initial_position[0]) * time),
                         np.sin((np.pi*v/initial_position[0]) * time)
                         ])
    
    """
    to be calculated by the taking the derivative:  (a)
    vel = v * np.array([0.5*np.cos(w*periodInterval), np.cos(2*w*periodInterval)])
    acc = -q * np.array([0.25*np.sin(w*periodInterval),np.sin(2*w*periodInterval)])
    """
    
    
    """
    Well be implemented later to gather a vector state describing position, velocity and acceleration 
    
    def getStateX(self):
        X = np.hstack((np.hstack((self.r.transpose(),self.vel.transpose())),self.acc.transpose()))
        return X
    """
    
moving_object = MovingObject()
position = moving_object.position

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


plt.show()
