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
    def collocate_state(self, states, controls, times, *params):

        # sanity
        assert len(states.shape) == len(controls.shape)
        assert len(times.shape) == 1
        assert states.shape[0] == controls.shape[0] == times.shape[0]

        # state collocation — https://en.wikipedia.org/wiki/Trapezoidal_rule
        f = vmap(lambda state, control: self.state_dynamics(state, control, *params))
        fa = f(states[1:,:], controls[1:,:])
        fb = f(states[:-1,:], controls[:-1,:])
        dt = times[1:] - times[:-1]
        e = states[:-1,:] + dt.dot(fa + fb)/2.0 - states[1:,:]
        return e

    @partial(jit, static_argnums=(0,))
    def collocate_costate(self, states, costates, controls, times, homotopy, *params):

        # sanity
        assert len(states.shape) == len(costates.shape) == len(controls.shape)
        assert len(times.shape) == 1
        assert states.shape[0] == states.shape[0] == costates.shape[0] == controls.shape[0] == times.shape[0]

        # costate_collocation
        f = vmap(lambda state, costate, control: self.costate_dynamics(state, costate, control, homotopy, *params))
        fa = f(states[1:,:], costates[1:,:], controls[1:,:])
        fb = f(states[:-1,:], costates[:-1,:], controls[:-1,:])
        dt = times[1:] - times[:-1]
        e = costates[:-1,:] + dt.dot(fa + fb)/2.0 - costates[1:,:]
        return e

    def solve_indirect(self, states, costates, controls, T, homotopy, boundaries):
        # z = [T, s0, l0, u0, ..., sn, ln, un]

        # system parameters
        params = self.params.values()

        # number of collocation nodes
        n = states.shape[0]

        # compile decision vector bounds
        def get_bounds():
            zl = np.hstack((self.state_lb, -100*np.ones_like(self.state_lb), self.control_lb))
            zl = np.tile(zl, n)
            zl = np.hstack(([0.0], zl))
            zu = np.hstack((self.state_ub, 100*np.ones_like(self.state_ub), self.control_ub))
            zu = np.tile(zu, n)
            zu = np.hstack(([T], zu))
            return zl, zu
        self.get_bounds = jit(get_bounds)

        zl, zu = self.get_bounds()
        zr = uniform(PRNGKey(0), (zl.shape[0],), minval=zl, maxval=zu)

        # compile decision vector translator
        @jit
        def unflatten(z):
            T = z[0]
            z = z[1:]
            z = z.reshape(-1, self.state_dim*2 + self.control_dim)
            states = z[:,:self.state_dim]
            costates = z[:,self.state_dim:self.state_dim*2]
            controls = z[:,self.state_dim*2:self.state_dim*2+self.control_dim]
            return states, costates, controls, T

        # gradient of Hamiltonian wrt control = 0
        @vmap
        @jit
        def hamiltonian_jac_control(state, costate, control):
            return jacfwd(self.hamiltonian, argnums=2)(state, costate, control, homotopy, *params)

        # compile fitness vector
        print('Compiling fitness vector...')
        def fitness(z):

            # translate decision vector
            states, costates, controls, T = unflatten(z)

            # time grid
            n = states.shape[0]
            times = np.linspace(0, T, n)

            # collocation constraints
            e0 = self.collocate_state(states, controls, times, *params).flatten()
            e1 = self.collocate_costate(states, costates, controls, times, homotopy, *params).flatten()
            e = np.hstack((e0, e1))

            # boundary constraints
            e1, e2 = boundaries(states[0,:], costates[0,:], states[-1,:], costates[-1,:])
            e = np.hstack((e, e1, e2))

            # Jacobian of Hamiltonian wrt control = 0
            e0 = hamiltonian_jac_control(states, costates, controls)
            e = np.hstack((e, e0.flatten()))**2
            
            # fitness vector — only constraints
            return np.hstack(([1], e))

        self.fitness = jit(fitness)

        # compile sparse Jacobian
        print('Compiling Jacobian...')
        gradient = jit(jacfwd(self.fitness))
        gid = np.vstack((np.nonzero(gradient(zr)))).T
        self.gradient = jit(lambda z: gradient(z)[[*gid.T]])
        self.gradient_sparsity = jit(lambda : gid)

        # fitness dimensions
        self.get_nobj = jit(lambda: 1)
        nec = len(self.fitness(zr)) - 1
        self.get_nec = jit(lambda: nec)

        # pygmo problem
        print('Optimising...')
        prob = pg.problem(udp=self)
        algo = pg.ipopt()
        algo.set_integer_option('max_iter', 2000)
        algo = pg.algorithm(algo)
        algo.set_verbosity(1)
        pop = pg.population(prob=prob, size=1)
        pop = algo.evolve(pop)
        z = pop.champion_x
        np.save('decision.npy', z)

        states, costates, control, T = unflatten(z)
        self.plot('../img/pontryagin.png', states, dpi=5000)


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
    n = 20
    k = PRNGKey(0)
    states = uniform(k, (n, system.state_dim), minval=system.state_lb, maxval=system.state_ub)
    costates = normal(k, (n, system.state_dim))
    controls = uniform(k, (n, system.control_dim), minval=system.control_lb, maxval=system.control_ub)
    times = np.linspace(0.0, 100.0, num=n)
    homotopy = [0.0, 0.0]
    T = 10.0

    # boundary constraints
    @jit
    def boundaries(state0, costate0, statef, costatef):
        e0 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        e0 -= state0 
        e1 = np.array([100, 100, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        e1 -= statef
        return e0, e1

    res = system.solve_indirect(states, costates, controls, T, homotopy, boundaries)
    print(res)