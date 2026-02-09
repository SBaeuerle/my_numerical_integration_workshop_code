import numpy as np

def damped_pendulum_ode(t: float, z: np.ndarray):
    g = 9.81
    L = 0.5
    m = 0.2
    d = 0.2

    omega_0: float = np.sqrt(g / L)
    D: float = d / (2 * m * omega_0)

    theta, dtheta = z
    return np.array([dtheta, -2*D*omega_0*dtheta - omega_0**2*np.sin(theta)])