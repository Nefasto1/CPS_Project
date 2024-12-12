import numpy as np

from scipy.optimize import minimize

from src.simulation import simulation

class PID_Controller():
    def __init__(self, car, obstacles, target, reference=5, simulation_time=200, dt=0.5, inertia=0.8, kalman=True, noise=[1, 1, 0.5, 0.5], LQR=True):
        """
        Initialize the PID Controller.

        Parameters
        ----------
        car: The list with the car informations
        obstacles: The list with the obstacles informations
        target: The list with the target informations
        reference: The reference module speed to reach
        simulation_time: The simulation time limit
        dt: The used interval time
        inertia: The inertia constant for the speed
        kalman: A boolean for use or not the kalman filter in the simulation (if False use the Luemberger observer)
        noise: List of noise values for both process and measurement
        """
        self.dt              = dt
        self.inertia         = inertia
        self.simulation_time = simulation_time
        self.reference       = reference
        
        self.kalman          = kalman
        self.noise           = noise
        self.LQR             = LQR

        self.car       = car
        self.obstacles = obstacles
        self.target    = target

        self.reference_history = np.ones((1, round(simulation_time / dt)))
        
        self.reset()

    def reset(self):
        """
        Reset the PID and history
        """
        self.prev_errors   = []
        # self.prev_errors   = [0]

        self.speed_history = []
        self.u_history     = []

    def control(self, reference, predicted, Kp, Kd, Ki):
        """
        Evaluate the input given the current prediction

        Parameters
        ----------
        reference: The reference signal to use (Method accessible from external, to fix)
        predicted: The predicted new signal
        Kp: Proportional Gain
        Ki: Integral Gain
        Kd: Derivative Gain

        Returns
        -------
        u: The new input
        """
        # Evaluate the error and the values for the integrator and the derivator
        error    = np.array([reference - predicted]).flatten()

        integral = (np.sum(self.prev_errors[:50], axis=0) + error) * self.dt
        # integral = (np.sum(self.prev_errors[:50]) + error) * self.dt
        integral = np.clip(integral, -1.5, 1.5)

        derivative     = ( error - self.prev_errors[-1] ) / self.dt if len(self.prev_errors) > 0 else 0 

        # Evaluate the input
        u = Kp * error    \
          + Ki * integral \
          + Kd * derivative    

        # Save the error for the next derivative evaluation
        self.prev_errors = np.vstack((self.prev_errors, error)) if len(self.prev_errors) != 0 else [error]
        # self.prev_errors.append(error)

        # Take the absolute value (we work with the speed module)
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

        # Rising Time evaluation
        rising_time_idx   = np.zeros(reference_objective_intersection.shape[0]) - 1

        rised = reference_objective_intersection.any(1)
        if rised.any():
            rising_time_idx   = reference_objective_intersection[rised].argmax(1) 

        rising_time_error = np.zeros(reference_objective_intersection.shape[0])
        if (rising_time_idx != -1).any():
            rising_time_error[rising_time_idx != -1] = time[rising_time_idx[rising_time_idx != -1]]

        # Steady State evaluation
        steady_state_error = np.abs(speed_reference_differences[:, -1])

        # Overshooting evaluation
        overshooting_error = maximum_speed - references[:, -1]
        
        return rising_time_error, overshooting_error, steady_state_error

    def cost_function(self, K):
        """
        Cost function related to the Gain parameters

        Parameters
        ----------
        K: Tuple composed by (Proportional, Integral, Derivative) Gain

        Returns
        -------
        cost: The cost of using the input Gain parameters
        """
        Kp, Ki, Kd = K
        # Reset the simulation
        self.reset()

        # Retrive the speed history from the simulation and take the modules
        _, self.speed_history, _, _, _, _, _, _, _ = simulation(self.car, self.obstacles, self.target, self, 0.5, 0.8, self.reference, self.simulation_time, Kp, Ki, Kd, kalman=self.kalman, noise=self.noise, LQR=self.LQR)
        self.speed_history = np.sqrt((self.speed_history**2).sum(1, keepdims=True))

        # Evaluate the errors
        rising_time_error, overshooting_error, steady_state_error = self.performance_meas()

        # Evaluate the costs
        cost = rising_time_error     \
             + overshooting_error  \
             + steady_state_error

        return cost.sum()
        
    def optimize_pid(self, initial_guess):
        """
        Method to optimize the Gain parameters

        Parameters
        ----------
        initial_guess: Tuple composed by (Proportional, Integral, Derivative) Gain
        
        Returns
        -------
        K: Tuple composed by the optimized (Proportional, Integral, Derivative) Gain
        """
        # Minimize the cost function to optimize the PID
        optimized_values = minimize(self.cost_function, initial_guess, method='Powell', options={'disp': False})

        return optimized_values.x