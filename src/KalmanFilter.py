import numpy as np

class KalmanFilter:
    def __init__(self, A, B, C, Q, R, Ts):
        """
        Initialize the Kalman Filter.
        
        Parameters:
        A (ndarray): State transition matrix.
        B (ndarray): Control input matrix.
        C (ndarray): Measurement matrix.
        Q (ndarray): Process noise covariance matrix.
        R (ndarray): Measurement noise covariance matrix.
        Ts (float): Sampling time for discretization.
        """
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.Ts = Ts

    def predict(self, x_hat, P, u):
        """
        Performs the prediction step of the Kalman filter.
        
        Parameters:
        x_hat (ndarray): Prior state estimate.
        P (ndarray): Prior error covariance matrix.
        u (ndarray): Control input.
        
        Returns:
        x_pred (ndarray): Predicted state estimate.
        P_pred (ndarray): Predicted error covariance matrix.
        """
        # x_pred = self.Ad @ x_hat + self.Bd @ u
        # P_pred = self.Ad @ P @ self.Ad.T + self.Q
        x_pred = self.A @ x_hat + self.B @ u
        P_pred = self.A @ P @ self.A.T + self.Q
        
        return x_pred, P_pred

    def update(self, x_pred, P_pred, y):
        """
        Performs the update step of the Kalman filter.
        
        Parameters:
        x_pred (ndarray): Predicted state estimate.
        P_pred (ndarray): Predicted error covariance matrix.
        y (ndarray): Measurement vector.
        
        Returns:
        x_updated (ndarray): Updated state estimate.
        P_updated (ndarray): Updated error covariance matrix.
        """
        # Innovation (measurement residual)
        z = y - self.C @ x_pred

        # Innovation covariance
        S = self.R + self.C @ P_pred @ self.C.T

        # Optimal Kalman gain
        K = P_pred @ self.C.T @ np.linalg.inv(S)

        # Updated state estimate and covariance
        x_updated = x_pred + K @ z
        P_updated = (np.eye(P_pred.shape[0]) - K @ self.C) @ P_pred
        
        return x_updated, P_updated

    def filter_step(self, x_hat, P, u, y):
        """
        Performs a full prediction-update step of the Kalman filter.
        
        Parameters:
        x_hat (ndarray): Prior state estimate.
        P (ndarray): Prior error covariance matrix.
        u (ndarray): Control input.
        y (ndarray): Measurement vector.
        
        Returns:
        x_updated (ndarray): Updated state estimate.
        P_updated (ndarray): Updated error covariance matrix.
        """
        x_pred, P_pred       = self.predict(x_hat, P, u)
        x_updated, P_updated = self.update(x_pred, P_pred, y)
        
        return x_updated, P_updated