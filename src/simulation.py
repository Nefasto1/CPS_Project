from src.Car import Car
from src.Obstacle import Obstacle
from src.LQRController import LQRController
from src.KalmanFilter import KalmanFilter
from src.LuenbergerObserver import LuenbergerObserver
from src.collisions import get_new_targets

import numpy as np

def define_system(dt=0.5, inertia=0.8):
    """
    Function to define the linear system based just on the time and inertia constants

    Parameters
    ----------
    dt: Sampling time for discretization
    inertia: Inertia of the speed

    Returns
    -------
    A: State transition matrix.
    B: Control input matrix.
    C: Measurement matrix.
    Q: Process noise covariance matrix.
    R: Measurement noise covariance matrix.
    """
    # Define system dynamics (A, B matrices)
    A = np.array([[1, 0, inertia*dt, 0],
                  [0, 1, 0, inertia*dt],
                  [0, 0, inertia, 0],
                  [0, 0, 0, inertia]])  
    
    B = np.array([[dt**2, 0],
                  [0, dt**2],
                  [dt, 0],
                  [0, dt],])  
    
    C = np.diag([1, 1, 1, 1])
    
    # Define cost matrices (Q, R)
    Q = np.diag([1, 1, 1e5, 1e5])  
    R = np.diag([1e5, 1e5])  

    return A, B, C, Q, R
    
def simulation(car, obstacles, targets, PID=None, dt=0.5, inertia=0.8, reference=5, simulation_time=200, Kp=None, Ki=None, Kd=None, kalman=True, noise=[1, 1, 0.5, 0.5]):
    """
    Function to simulate the system

    Parameters
    ----------
    car: Car's initial informations ((x, y), (speed_x, speed_y), theta)
    obstacles: List of obstacles informations ((x, y), theta)
    targets: Target state informations (x, y, speed_x, speed_y)
    PID: Initialized PIDController object or None
    dt: Sampling time for discretization
    inertia: Inertia of the speed
    reference: Reference signal for PID controller
    simulation_time: Total simulation time
    Kp: Proportional Gain
    Ki: Integral Gain
    Kd: Derivative Gain
    kalman: Boolean, If True uses the Kalman Filter for state estimation, If False the Luenberger Observer will be used
    noise: List of noise, the same for process and measurement (x_noise, y_noise, speed_x_noise, speed_y_noise)

    Returns
    -------
    states: List of the car's states
    u_list: List of the car's inputs
    target_list: List of the car's unique targets
    """
    # Take the car informations
    initial_coords = car[0]
    initial_speeds = car[1]
    initial_theta  = car[2]

    # Entity definition
    car = Car(initial_coords, initial_speeds, initial_theta, noise=noise)
    obstacles = [Obstacle(coords, angle) for coords, angle in obstacles]

    # LQR Definition
    A, B, C, Q, R  = define_system(dt, inertia)
    lqr_controller = LQRController(A, B, Q, R, dt)

    # Initialize the process and measurement
    process_covariance      = np.diag(noise)
    measurement_covariance  = np.diag(noise)

    # Kalman Filter Initialization 
    kalman_filter   = KalmanFilter(A, B, C, process_covariance, measurement_covariance)
    predicted_state = car.get_state()
    
    # Luemberger Observer Initialization 
    luenberger      = LuenbergerObserver(A, B, C, np.array([-1e-10]*4), predicted_state)
    P = np.zeros_like(C)

    # Time Scheduling for multitarget system
    target_round_change = ((simulation_time // dt)//len(targets))
    counter = 1

    # Initialize the history lists
    u_list      = []
    states      = [car.get_state()]
    target_list = []

    # For all the discretized time-steps
    for i in range(round(simulation_time / dt)):
        # Determine the target (for multitarget system)
        target_idx = i//target_round_change
        new_target = targets[int(target_idx)]

        # Counter to avoid collapse to temporaneous target
        counter -= 1
        if counter == 0:
            # If close to an obstacle in front find a temporaneous target to avoid it
            new_target, counter = get_new_targets(predicted_state, obstacles, new_target)

        # Find the target direction
        diff = predicted_state - new_target
    
        # LQR for track optimization
        u = lqr_controller.compute_control_input(diff)

        u = np.clip(u, -5, 5)  # Apply saturation limit to control input
        if PID is not None:
            # PID for cruise control
            theta            = np.arctan2(u[1], u[0])
            
            module           = np.sqrt((u ** 2).sum())
            corrected_module = PID.control(reference, module, Kp, Ki, Kd)
        
            u = corrected_module * np.cos(theta), corrected_module * np.sin(theta)
            
        u = np.clip(u, -5, 5)  # Apply saturation limit to control input

        # Perform a step
        # speed = car.accelerate(u)
        speed = car.accelerate_noisy(u)
        measurement = car.get_state_noisy()

        # Estimate the state
        if kalman:
            predicted_state, P = kalman_filter.filter_step(predicted_state, P, u, measurement)
        else:
            predicted_state = luenberger.update(u, measurement)
        
        # Append the results
        states.append(car.get_state())
        u_list.append(u)
        target_list.append(new_target)
    
    states      = np.array(states)
    u_list      = np.array(u_list)
    target_list = np.array(target_list)

    return states, u_list, np.unique(target_list, axis=0)