from src.Car import Car
from src.Obstacle import Obstacle
from src.LQRController import LQRController
from src.KalmanFilter import KalmanFilter
from src.LuenbergerObserver import LuenbergerObserver
from src.collisions import get_new_targets, check_collision
from src.utils import get_modules, get_directional, window_mean, draw

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
    Q = np.diag([0, 0, 100, 100])  
    R = np.diag([10, 10])  

    return A, B, C, Q, R
    
def simulation(car, obstacles, targets, PID=None, dt=0.5, inertia=0.8, reference=5, simulation_time=200, Kp=None, Ki=None, Kd=None, kalman=True, noise=[1, 1, 0.5, 0.5], LQR=True):
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

    if PID is not None:
        PID.reset()

    # Time Scheduling for multitarget system
    target_idx = 0
    counter    = 1

    # Initialize the history lists
    u_list         = []
    states         = [car.get_state()]
    target_list    = []
    reference_list = []
    theta_list     = [0]
    frames         = []
    has_completed   = []
    has_collided   = []

    # For all the discretized time-steps
    for i in range(round(simulation_time / dt)):
        completed = False
        collided  = False
        
        # Determine the target (for multitarget system)
        target_diff = predicted_state - targets[target_idx]
        target_dist = get_modules(target_diff[:2])
        target_module = get_modules(target_diff[2:])
        if target_dist < 25 and target_module < 0.25:
            if target_idx == len(targets) - 1:
                completed = True
            else:
                target_idx += 1

        # Determine if has collided
        if check_collision(car, obstacles):
            collided = True
            
        new_target = targets[target_idx]

        # Counter to avoid collapse to temporaneous target
        counter -= 1
        if counter == 0 and len(obstacles) > 0:
            # If close to an obstacle in front find a temporaneous target to avoid it
            new_target, counter = get_new_targets(predicted_state, obstacles, new_target)

        # Find the target direction
        diff = predicted_state - new_target
        
        diff[:2] = np.clip(diff[:2], -1, 1)
        diff[2:] = np.clip(diff[2:], -0.5, 0.5)

        reference = diff[2:]
        # LQR for track optimization
        if LQR:
            u = lqr_controller.compute_control_input(diff)
        else:
            u = -diff[:2]
    
        if PID is not None:            
            corrected_module = PID.control(reference, u, Kp, Ki, Kd)

        _, _, speed_x, speed_y = car.get_state()
        theta = np.arctan2(speed_y, speed_x)
        

        u = window_mean(u_list+[u])
        u = np.clip(u, -1, 1)
        
        # Perform a step
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
        reference_list.append(reference)
        theta_list.append(theta)
        if not new_target.tolist() in targets.tolist():
            target_list.append(new_target)

        has_collided.append(collided)
        has_completed.append(completed)

        # frame = draw(car, obstacles)
        # frames.append(frame)

    states      = np.array(states)
    u_list      = np.array(u_list)
    target_list = np.array(target_list)

    return states, u_list, np.unique(target_list, axis=0), reference_list, theta_list, has_completed, has_collided, frames