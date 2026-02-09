import numpy as np
import math

def euler_explicit(fcn, t_interval: list, z0: list, h: float):

    # Allocate
    t_start = min(t_interval)
    t_end = max(t_interval)

    # Init data container
    t_container: list = [t_start]
    u_container: list = [z0]

    number_of_iterations = math.ceil((t_end-t_start)/h)

    # Assign initial conditions
    u_i = np.array(z0)
    t_i = t_start

    # Euler explicit
    for i in range(1, number_of_iterations):
        # Compute
        t_ip1 = t_start + i * h
        u_ip1 = u_i + h * fcn(t_i,u_i)
        
        # Update
        t_i = t_ip1
        u_i = u_ip1
        
        #save
        t_container.append(t_ip1)
        u_container.append(u_ip1)


    return np.array(t_container), np.array(u_container)