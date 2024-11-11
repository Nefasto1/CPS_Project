import numpy as np

from scipy.optimize import minimize

from src.simulation import simulation

class PID_Controller():
    def __init__(self, car, obstacles, target, reference=5, simulation_time=200, dt=0.5, inertia=0.8, kalman=True):
        self.dt              = dt
        self.inertia         = inertia
        self.simulation_time = simulation_time
        self.reference       = reference
        self.kalman          = kalman

        self.car       = car
        self.obstacles = obstacles
        self.target    = target

        self.reference_history = np.ones((1, round(simulation_time / dt)))
        
        self.reset()

    def reset(self):
        self.integral   = 0
        self.prev_error = 0

        self.speed_history     = []
        self.u_history         = []

    def control(self, reference, predicted, Kp=None, Kd=None, Ki=None):
        if Kp is None or Kd is None or Ki is None:
            Kp, Kd, Ki = self.K
        
        error          = reference - predicted
        self.integral += error * self.dt
        derivative     = ( error - self.prev_error ) / self.dt

        u = Kp * error         \
          + Ki * self.integral \
          + Kd * derivative    

        self.prev_error = error
        
        return np.abs(u)
        
    def performance_meas(self):
        """Evaluate motor performance based on rise time, overshoot, and steady-state error."""
        time       = np.arange(len(self.speed_history)) * self.dt
        speeds     = np.array(self.speed_history).T
        references = np.array(self.reference_history).T

        # Utility values
        maximum_speed                    = speeds.max(1)
        speed_reference_differences      = speeds - references
        reference_objective_intersection = np.abs(speed_reference_differences) < 0.1

        # Rising Time
        rising_time_idx   = np.zeros(reference_objective_intersection.shape[0]) - 1

        rised = reference_objective_intersection.any(1)
        if rised.any():
            rising_time_idx   = reference_objective_intersection[rised].argmax(1) 

        rising_time_error = np.zeros(reference_objective_intersection.shape[0])
        if (rising_time_idx != -1).any():
            rising_time_error[rising_time_idx != -1] = time[rising_time_idx[rising_time_idx != -1]]

        # Steady State
        steady_state_error = np.abs(speed_reference_differences[:, -1])

        # Overshooting
        overshooting_error = maximum_speed - references[:, -1]
        
        return rising_time_error, overshooting_error, steady_state_error

    def cost_function(self, K):
        Kp, Ki, Kd = K
        self.reset()
        
        _, self.speed_history, _ = simulation(self.car, self.obstacles, self.target, self, 0.5, 0.8, self.reference, self.simulation_time, Kp, Ki, Kd, kalman=self.kalman)
        self.speed_history = np.sqrt((self.speed_history**2).sum(1, keepdims=True))

        rising_time_error, overshooting_error, steady_state_error = self.performance_meas()

        cost = rising_time_error     \
             + overshooting_error*5  \
             + steady_state_error*10

        return cost.sum()
        
    def optimize_pid(self, initial_guess):
        optimized_values = minimize(self.cost_function, initial_guess, method='Powell', options={'disp': True})

        self.K = optimized_values.x

        return self.K