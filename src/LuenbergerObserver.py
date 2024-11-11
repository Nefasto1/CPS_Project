import numpy as np
from scipy.signal import place_poles

class LuenbergerObserver:
    def __init__(self, A, B, C, poles, x_hat):
        """
        Initialize the Luenberger Observer.
        
        Parameters:
        A (numpy.ndarray): System dynamics matrix.
        B (numpy.ndarray): Control input matrix.
        C (numpy.ndarray): Output matrix.
        L (numpy.ndarray): Observer gain matrix.
        dt (float): Time step for the observer updates.
        """
        A[:, 2:] *= 0.5
        self.A = A
        self.B = B
        self.C = C
        self.x_hat = x_hat

        # Define desired poles (negative real values for stable observer)
        pole_placement = place_poles(A, C, poles)
        self.L = pole_placement.gain_matrix

    def update(self, u, y):
        """
        Update the observer state estimate.
        
        Parameters:
        u (numpy.ndarray): Control input vector at the current time step.
        y (numpy.ndarray): Measurement vector at the current time step.
        
        Returns:
        numpy.ndarray: Updated state estimate.
        """
        # Compute the output estimation error
        y_hat = self.C @ self.x_hat
        e_y = y - y_hat  # Measurement error

        # Luenberger observer update
        x_hat_dot = self.A @ self.x_hat + self.B @ u + e_y @ self.L
        self.x_hat = x_hat_dot.reshape(-1)
        
        return self.x_hat
