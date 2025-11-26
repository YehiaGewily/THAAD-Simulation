# environment.py
import numpy as np

class Environment:
    def __init__(self):
        self.g = 9.81  # Gravity (m/s^2)
        self.rho_0 = 1.225  # Sea level air density (kg/m^3)
        self.scale_height = 8500  # Scale height for atmosphere (m)

    def get_air_density(self, altitude):
        if altitude < 0:
            return self.rho_0
        # Exponential atmosphere model
        return self.rho_0 * np.exp(-altitude / self.scale_height)

    def get_gravity_vector(self):
        # Downward gravity in Z axis
        return np.array([0.0, 0.0, -self.g])