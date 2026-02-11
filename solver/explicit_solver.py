import numpy as np
import math

from matplotlib.pyplot import savefig


def euler_explicit(fcn, t_interval: list, z0: np.ndarray, h: float):

    # Time and iterations
    t_start = min(t_interval)
    t_end = max(t_interval)
    number_of_iterations = math.ceil((t_end - t_start) / h) - 1

    # Assign initial conditions
    u_i = z0
    t_i = t_start

    # Init data container
    t_container: list = [t_start]
    u_container: list = [u_i]

    # Euler explicit
    for i in range(0, number_of_iterations):
        # Compute
        t_ip1 = t_start + (i+1) * h
        u_ip1 = u_i + h * fcn(t_i,u_i)
        
        # Update
        t_i = t_ip1
        u_i = u_ip1
        
        #save
        t_container.append(t_ip1)
        u_container.append(u_ip1)

    return np.array(t_container), np.array(u_container)

def mid_point_rule(fcn, t_interval: list, z0: np.ndarray, h:float):

    # Time and iterations
    t_start = min(t_interval)
    t_end = max(t_interval)
    number_of_iterations = math.ceil((t_end - t_start) / h) - 1

    # Assign initial conditions
    t_i = t_start
    u_i = z0

    # init data container
    t_container: list = [t_start]
    u_container: list = [u_i]

    # Mid-point rule
    for i in range(0, number_of_iterations):

        t_ip1 = t_start + (i+1) * h
        k1 = fcn(t_i,u_i)
        k2 = fcn(t_i + 0.5*h, u_i + 0.5*h*k1)
        u_ip1 = u_i + h*k2

        #save
        t_container.append(t_ip1)
        u_container.append(u_ip1)

        #iterate
        u_i = u_ip1
        t_i = t_ip1

    return np.array(t_container), np.array(u_container)


def RK4(fcn, t_interval: list, z0: np.ndarray, h: float):

    # Time and iterations
    t_start = min(t_interval)
    t_end = max(t_interval)
    number_of_iterations = math.ceil((t_end-t_start)/h)-1

    # Assign initial conditions
    u_i = z0
    t_i = t_start

    # Initialize data containers
    t_container : list = [t_start]
    u_container: list = [u_i]

    for i in range(0, number_of_iterations):

        t_ip1 = t_start + (i+1) * h
        k1 = fcn(t_i, u_i)
        k2 = fcn(t_i + 0.5 * h, u_i + 0.5 * h * k1)
        k3 = fcn(t_i + 0.5 * h, u_i + 0.5 * h * k2)
        k4 = fcn(t_i + h, u_i + h * k3)
        u_ip1 = u_i + 1/6 * h * (k1 + 2 * k2 + 2 * k3 + k4)

        # save
        t_container.append(t_ip1)
        u_container.append(u_ip1)

        #iterate
        t_i = t_ip1
        u_i = u_ip1

    return np.array(t_container), np.array(u_container)

