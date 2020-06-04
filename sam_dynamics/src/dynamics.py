# General class for dynamics
# Use, e.g., for optimal control, MPC, etc.
# Christopher Iliffe Sprague

import numpy as np, matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Dynamics:

    def eom(self, state, control):
        raise NotImplementedError

    def eom_jac(self, state, controls):
        raise NotImplementedError

    def propagate(self, state, controller, t0, tf, atol=1e-8, rtol=1e-8, method='DOP853'):

        # integrate dynamics
        sol = solve_ivp(
            lambda t, x: self.eom(x, controller(x)),
            (t0, tf),
            state,
            method=method,
            rtol=rtol,
            atol=atol,
            jac=lambda t, x: self.eom_jac(x, controller(x))
        )

        # return times, states, and controls
        times, states = sol.t, sol.y.T
        controls = np.apply_along_axis(controller, 1, states)
        return times, states, controls

    def plot(self, states, controls=None):
        raise NotImplementedError
