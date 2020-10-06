# General class for dynamics
# Use, e.g., for optimal control, MPC, etc.
# Christopher Iliffe Sprague

import jax.numpy as np
from jax import jit, jacfwd, hessian, vmap
from jax.experimental.ode import odeint
from jax.random import normal, uniform, PRNGKey
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial
import pygmo as pg

class Dynamics:

    def __init__(self, **kwargs):

        # constant parameters from child class
        assert all(key in self.params.keys() for key in kwargs.keys())
        self.params.update(kwargs)
        
    @staticmethod
    def lagrangian(state, control, homotopy, *params):
        raise NotImplementedError('Implement lagrangian in child class.')

    @staticmethod
    def state_dynamics(state, control, *params):
        raise NotImplementedError('Implement state_dynamics in child class.')

    @partial(jit, static_argnums=(0,))
    def state_dynamics_jac_state(self, state, control, *params):
        return jacfwd(self.state_dynamics)(state, control, *params)

    @partial(jit, static_argnums=(0,))
    def hamiltonian(self, state, costate, control, homotopy, *params):
        f = self.state_dynamics(state, control, *params)
        L = self.lagrangian(state, control, homotopy, *params)
        H = costate.dot(f) + L
        return H

    @partial(jit, static_argnums=(0,))
    def costate_dynamics(self, state, costate, control, homotopy, *params):
        return -jacfwd(self.hamiltonian)(state, costate, control, homotopy, *params)

    @partial(jit, static_argnums=(0,))
    def collocate_lagrangian(self, states, controls, times, costs, homotopy, *params):

        # sanity
        assert len(states.shape) == len(controls.shape) == 2
        assert len(times.shape) == len(costs.shape) == 1
        assert states.shape[0] == controls.shape[0] == times.shape[0] == costs.shape[0]

        # https://en.wikipedia.org/wiki/Trapezoidal_rule
        f = vmap(lambda state, control: self.lagrangian(state, control, homotopy, *params))
        fa = f(states[:-1,:], controls[:-1,:])
        fb = f(states[1:,:], controls[1:,:])
        dt = times[1:] - times[:-1]
        e = costs[:-1] + dt*(fa + fb)/2.0 - costs[1:]
        return e

    @partial(jit, static_argnums=(0,))
    def collocate_state(self, states, controls, times, *params):

        # sanity
        assert len(states.shape) == len(controls.shape) == 2
        assert len(times.shape) == 1
        assert states.shape[0] == controls.shape[0] == times.shape[0]

        # https://en.wikipedia.org/wiki/Trapezoidal_rule
        f = vmap(lambda state, control: self.state_dynamics(state, control, *params))
        fa = f(states[:-1,:], controls[:-1,:])
        fb = f(states[1:,:], controls[1:,:])
        dt = times[1:] - times[:-1]
        e = states[:-1,:] + dt.dot(fa + fb)/2.0 - states[1:,:]
        return e

    def solve_direct(self, states, controls, T, homotopy, boundaries):

        # sanity
        assert states.shape[0] == controls.shape[0]
        assert states.shape[1] == self.state_dim
        assert controls.shape[1] == self.control_dim

        # system parameters
        params = self.params.values()

        # number of collocation nodes
        n = states.shape[0]
        
        # decision vector bounds
        @jit
        def get_bounds():
            zl = np.hstack((self.state_lb, self.control_lb))
            zl = np.tile(zl, n)
            zl = np.hstack(([0.0], zl))
            zu = np.hstack((self.state_ub, self.control_ub))
            zu = np.tile(zu, n)
            zu = np.hstack(([np.inf], zu))
            return zl, zu

        # decision vector maker
        @jit
        def flatten(states, controls, T):
            z = np.hstack((states, controls)).flatten()
            z = np.hstack(([T], z))
            return z

        # decsision vector translator
        @jit
        def unflatten(z):
            T = z[0]
            z = z[1:].reshape(n, self.state_dim + self.control_dim)
            states = z[:,:self.state_dim]
            controls = z[:,self.state_dim:]
            return states, controls, T

        # fitness vector
        print('Compiling fitness...')
        @jit
        def fitness(z):

            # translate decision vector
            states, controls, T = unflatten(z)

            # time grid
            n = states.shape[0]
            times = np.linspace(0, T, n)

            # objective
            L = vmap(lambda state, control: self.lagrangian(state, control, homotopy, *params))
            L = L(states, controls)
            J = np.trapz(L, dx=T/(n-1))

            # Lagrangian state dynamics constraints, and boundary constraints
            # e0 = self.collocate_lagrangian(states, controls, times, costs, homotopy, *params)
            e1 = self.collocate_state(states, controls, times, *params)
            e2, e3 = boundaries(states[0,:], states[-1,:])
            e = np.hstack((e1.flatten(), e2, e3))**2

            # fitness vector
            return np.hstack((J, e))

        # z = flatten(states, controls, T)
        # fitness(z)

        # sparse Jacobian
        print('Compiling Jacobian and its sparsity...')
        gradient = jit(jacfwd(fitness))
        z = flatten(states, controls, T)
        sparse_id = np.vstack((np.nonzero(gradient(z)))).T
        sparse_gradient = jit(lambda z: gradient(z)[[*sparse_id.T]])
        gradient_sparsity = jit(lambda : sparse_id)
        print('Jacobian has {} elements.'.format(sparse_id.shape[0]))

        # assign PyGMO problem methods
        self.fitness = fitness
        self.gradient = sparse_gradient
        self.gradient_sparsity = gradient_sparsity
        self.get_bounds = get_bounds
        self.get_nobj = jit(lambda: 1)
        nec = fitness(z).shape[0] - 1
        self.get_nec = jit(lambda: nec)

        # plot before
        states, controls, T = unflatten(z)
        self.plot('../img/direct_before.png', states, dpi=1000)

        # solve NLP with IPOPT
        print('Solving...')
        prob = pg.problem(udp=self)
        algo = pg.ipopt()
        algo.set_integer_option('max_iter', 1000)
        algo = pg.algorithm(algo)
        algo.set_verbosity(1)
        pop = pg.population(prob=prob, size=0)
        pop.push_back(z)
        pop = algo.evolve(pop)

        # save and plot solution
        z = pop.champion_x
        np.save('decision.npy', z)
        states, controls, T = unflatten(z)
        self.plot('../img/direct_after.png', states, dpi=1000)

    def plot(self, states, controls=None):
        raise NotImplementedError

    def propagate(self, state, controller, t0, tf, atol=1e-8, rtol=1e-8, method='DOP853'):

        # integrate dynamics
        sol = solve_ivp(
            jit(lambda t, x: self.state_dynamics(x, controller(x), *self.params.values())),
            (t0, tf),
            state,
            method=method,
            rtol=rtol,
            atol=atol,
            jac=jit(lambda t, x: self.state_dynamics_jac_state(x, controller(x), *self.params.values()))
        )

        # return times, states, and controls
        times, states = sol.t, sol.y.T
        controls = np.apply_along_axis(controller, 1, states)
        return times, states, controls

if __name__ == '__main__':

    from fossen import Fossen

    # instantiate Fossen model
    system = Fossen()
    params = system.params.values()

    # random states and controls
    n = 50
    k = PRNGKey(0)
    states = uniform(k, (n, system.state_dim), minval=system.state_lb, maxval=system.state_ub)
    costates = normal(k, (n, system.state_dim))
    controls = uniform(k, (n, system.control_dim), minval=system.control_lb, maxval=system.control_ub)
    costs = np.linspace(0, 100, n)
    times = np.linspace(0.0, 100.0, num=n)
    homotopy = [0.5, 0.0]
    T = 20.0

    # boundary constraints
    @jit
    def boundaries(state0, statef):
        e0 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        e0 -= state0 
        e1 = np.array([10, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        e1 -= statef
        return e0, e1

    system.solve_direct(states, controls, T, homotopy, boundaries)