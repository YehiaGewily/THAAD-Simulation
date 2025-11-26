# visualization.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

class Visualizer:
    def plot_simulation(self, threat_hist, interceptor_hist, intercept_point=None):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Convert list of arrays to simple numpy arrays for slicing
        th = np.array(threat_hist)
        ih = np.array(interceptor_hist)
        
        # Plot Threat (Red)
        if len(th) > 0:
            ax.plot(th[:,0], th[:,1], th[:,2], color='r', label='Ballistic Threat')
            ax.scatter(th[0,0], th[0,1], th[0,2], color='r', marker='o') # Start
        
        # Plot Interceptor (Blue)
        if len(ih) > 0:
            ax.plot(ih[:,0], ih[:,1], ih[:,2], color='b', label='Interceptor (THAAD)')
            ax.scatter(ih[0,0], ih[0,1], ih[0,2], color='b', marker='^') # Start

        # Plot Intercept
        if intercept_point is not None:
            ax.scatter(intercept_point[0], intercept_point[1], intercept_point[2], 
                       color='orange', s=200, marker='*', label='INTERCEPT')

        ax.set_xlabel('X Range (m)')
        ax.set_ylabel('Y Range (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title('Air Defense Simulation (3D)')
        ax.legend()
        plt.show()