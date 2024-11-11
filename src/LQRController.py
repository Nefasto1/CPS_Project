import numpy as np
from scipy.linalg import expm, solve_discrete_are

class LQRController:
    def __init__(self, A, B, Q, R, Ts):
        """
        Initialize the LQR Controller.
        
        Parameters:
        A (ndarray): State transition matrix of the system.
        B (ndarray): Control input matrix of the system.
        Q (ndarray): State cost matrix for LQR.
        R (ndarray): Control cost matrix for LQR.
        Ts (float): Sampling time for discretization.
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.Ts = Ts

        # Discretize the system matrices
        self.Ad = expm(A * Ts)
        self.Bd = np.linalg.pinv(A) @ (self.Ad - np.eye(A.shape[0])) @ B

        # Pre-compute the LQR gain matrix K
        self.K = self._compute_lqr_gain()

    def _compute_lqr_gain(self):
        """Computes the optimal LQR gain matrix using the Discrete Algebraic Riccati Equation."""
        P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        K = np.linalg.inv(self.Bd.T @ P @ self.Bd + self.R) @ (self.Bd.T @ P @ self.Ad)
        return K

    def compute_control_input(self, x):
        """
        Computes the control input using the LQR feedback law.
        
        Parameters:
        x (ndarray): Current state vector of the system.
        
        Returns:
        u (ndarray): Computed control input.
        """
        u = -self.K @ x
        return u
