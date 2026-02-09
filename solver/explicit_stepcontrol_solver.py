import numpy as np


def stepcontrol_mid_point_rule(fcn, t_interval, z0, h_init):

    # constant
    h_min = 1e-4
    h_max = 1.0

    max_rel_err = 1e-8
    max_abs_err = 1e-6

    # time
    t_start = min(t_interval)
    t_end = max(t_interval)

    # init data
    u_i = np.array(z0)
    t_i = t_start
    h_i = h_init

    # init container
    t_container: list = [t_i]
    u_container: list = [u_i]
    h_container: list = [h_i]
    error_container: list = [0]

    while t_i < t_end:

        h_i = min(h_i, t_end - t_i) # Do not overshoot the t_end

        t_ip1 = t_i + h_i
        k1 = fcn(t_i,u_i)
        k2 = fcn(t_i + 0.5*h_i, u_i + 0.5*h_i*k1)

        u_ip1_ee = u_i + h_i*k1
        u_ip1_mpr = u_i + h_i*k2

        # Error estimation
        diff = np.linalg.norm(u_ip1_mpr - u_ip1_ee)
        scale = max_abs_err + max_rel_err * np.linalg.norm(u_ip1_mpr)
        rel_err = diff / scale / np.sqrt(len(u_i))

        if rel_err > 1.0 and h_i > h_min: # step rejected
            h_i = max(h_i*0.5,h_min)
        else: # step accepted
            #save
            t_container.append(t_ip1)
            u_container.append(u_ip1_mpr)
            h_container.append(h_i)
            error_container.append(rel_err)

            t_i = t_ip1
            u_i = u_ip1_mpr

            if rel_err < 0.5:
                h_i = min(h_i*1.2,h_max)

    return np.array(t_container), np.array(u_container), np.array(h_container), np.array(error_container)
