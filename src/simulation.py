from src.Car import Car
from src.Obstacle import Obstacle
from src.LQRController import LQRController
from src.KalmanFilter import KalmanFilter
from src.LuenbergerObserver import LuenbergerObserver
from src.collisions import get_new_targets

import numpy as np

def define_system(dt=0.5, inertia=0.8):
    # Define system dynamics (A, B matrices)
    A = np.array([[1, 0, inertia*dt, 0],
                  [0, 1, 0, inertia*dt],
                  [0, 0, inertia, 0],
                  [0, 0, 0, inertia]])  # Example state matrix
    B = np.array([[dt**2, 0],
                  [0, dt**2],
                  [dt, 0],
                  [0, dt],])  # Example control matrix
    C = np.diag([1, 1, 1, 1])
    
    # Define cost matrices (Q, R)
    Q = np.diag([1, 1, 1e5, 1e5])  # Emphasizes position control more than speed
    R = np.diag([1e5, 1e5])  # Penalizes control effort

    return A, B, C, Q, R
    
def simulation(car, obstacles, targets, PID=None, dt=0.5, inertia=0.8, reference=5, simulation_time=200, Kp=None, Ki=None, Kd=None, kalman=True):
    initial_coords = car[0]
    initial_speeds = car[1]
    initial_theta  = car[2]

    # Entity definition
    car = Car(initial_coords, initial_speeds, initial_theta)
    obstacles = [Obstacle(coords, angle) for coords, angle in obstacles]

    # LQR Definition
    A, B, C, Q, R  = define_system(dt, inertia)
    lqr_controller = LQRController(A, B, Q, R, dt)

    process_covariance      = np.eye(C.shape[0])
    process_covariance[2:] *= 0.5
    measurement_covariance  = np.eye(C.shape[0])
    
    kalman_filter   = KalmanFilter(A, B, C, process_covariance, measurement_covariance, dt)
    predicted_state = car.get_state()
    
    luenberger      = LuenbergerObserver(A, B, C, np.array([-1e-10]*4), predicted_state)
    
    P = np.zeros_like(C)

    target_round_change = ((simulation_time // dt)//len(targets))
    counter = 1
    
    u_list      = []
    states      = [car.get_state()]
    target_list = []

    for i in range(round(simulation_time / dt)):
        counter -= 1
        target_idx = i//target_round_change
        new_target = targets[int(target_idx)]
            
        if counter == 0:
            # If close to an obstacle in front find a temporaneous target to avoid it
            new_target, counter = get_new_targets(predicted_state, obstacles, new_target)

        # Find the target direction
        diff = predicted_state - new_target
    
        # LQR for track optimization
        u = lqr_controller.compute_control_input(diff)
        u = np.clip(u, -1, 1)  # Apply saturation limit to control input

        if PID is not None:
            # PID for cruise control
            theta            = np.arctan2(u[1], u[0])
            
            module           = np.sqrt((u ** 2).sum())
            corrected_module = PID.control(reference, module, Kp, Ki, Kd)
        
            u = corrected_module * np.cos(theta), corrected_module * np.sin(theta)

        # Perform a step
        # speed = car.accelerate(u)
        speed = car.accelerate_noisy(u)
        measurement = car.get_state_noisy()

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