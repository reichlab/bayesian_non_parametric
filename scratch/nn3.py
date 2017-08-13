from theano import function, shared
from theano import tensor as T
import theano
import numpy as np
x = T.dscalar('x')
y = T.dscalar('y')



def sigmoid(x):
  return 1 / (1 + np.exp(-x))
def leapfrog(pos, vel, step):

    # from pos(t) and vel(t - eps/2), compute vel(t + eps / 2)
    dE_dpos = T.grad(energy_fn(pos).sum(), pos)
    new_vel = vel - step * dE_dpos

    # from vel(t + eps / 2) compute pos(t + eps)
    new_pos = pos + step * new_vel

    return [new_pos, new_vel],{}



def simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):


    def leapfrog(pos, vel, step):

        # from pos(t) and vel(t-stepsize/2), compute vel(t+stepsize/2)
        dE_dpos = T.grad(energy_fn(pos).sum(), pos)
        new_vel = vel - step * dE_dpos
        # from vel(t+stepsize/2) compute pos(t+stepsize)
        new_pos = pos + step * new_vel
        return [new_pos, new_vel], {}

    # compute velocity at time-step: t + stepsize/2
    initial_energy = energy_fn(initial_pos)
    dE_dpos = T.grad(initial_energy.sum(), initial_pos)
    vel_half_step = initial_vel - 0.5 * stepsize * dE_dpos

    # compute position at time-step: t + stepsize
    pos_full_step = initial_pos + stepsize * vel_half_step

    # perform leapfrog updates: the scan op is used to repeatedly compute
    # vel(t + (m-1/2)*stepsize) and pos(t + m*stepsize) for m in [2,n_steps].
    (all_pos, all_vel), scan_updates = theano.scan(leapfrog,
            outputs_info=[
                dict(initial=pos_full_step),
                dict(initial=vel_half_step),
                ],
            non_sequences=[stepsize],
            n_steps=n_steps - 1)
    final_pos = all_pos[-1]
    final_vel = all_vel[-1]

    # NOTE: Scan always returns an updates dictionary, in case the
    # scanned function draws samples from a RandomStream. These
    # updates must then be used when compiling the Theano function, to
    # avoid drawing the same random numbers each time the function is
    # called. In this case however, we consciously ignore
    # "scan_updates" because we know it is empty.
    assert not scan_updates

    # The last velocity returned by scan is vel(t +
    # (n_steps - 1 / 2) * stepsize) We therefore perform one more half-step
    # to return vel(t + n_steps * stepsize)
    energy = energy_fn(final_pos)
    final_vel = final_vel - 0.5 * stepsize * T.grad(energy.sum(), final_pos)

    # return new proposal state
    return final_pos, final_vel


def hmc_move(s_rng, positions, energy_fn, stepsize, n_steps):

    # sample random velocity
    initial_vel = s_rng.normal(size=positions.shape)

    # perform simulation of particles subject to Hamiltonian dynamics
    final_pos, final_vel = simulate_dynamics(
            initial_pos=positions,
            initial_vel=initial_vel,
            stepsize=stepsize,
            n_steps=n_steps,
            energy_fn=energy_fn)

    # accept/reject the proposed move based on the joint distribution
    accept = metropolis_hastings_accept(
            energy_prev=hamiltonian(positions, initial_vel, energy_fn),
            energy_next=hamiltonian(final_pos, final_vel, energy_fn),
            s_rng=s_rng)

    return accept, final_pos


def metropolis_hastings_accept(energy_prev, energy_next, s_rng):

    ediff = energy_prev - energy_next
    return (T.exp(ediff) - s_rng.uniform(size=energy_prev.shape)) >= 0


def kinetic_energy(vel):

    return 0.5 * (vel ** 2)


def hamiltonian(pos, vel, energy_fn):
    """
    Returns the Hamiltonian (sum of potential and kinetic energy) for the given
    velocity and position. Assumes mass is 1.
    """
    return energy_fn(pos) + kinetic_energy(vel)



def hmc_updates(positions, stepsize, avg_acceptance_rate, final_pos, accept,
                 target_acceptance_rate, stepsize_inc, stepsize_dec,
                 stepsize_min, stepsize_max, avg_acceptance_slowness):

    ## POSITION UPDATES ##
    # broadcast `accept` scalar to tensor with the same dimensions as
    # final_pos.
    accept_matrix = accept.dimshuffle(0, *(('x',) * (final_pos.ndim - 1)))
    # if accept is True, update to `final_pos` else stay put
    new_positions = T.switch(accept_matrix, final_pos, positions)

    ## STEPSIZE UPDATES ##
    # if acceptance rate is too low, our sampler is too "noisy" and we reduce
    # the stepsize. If it is too high, our sampler is too conservative, we can
    # get away with a larger stepsize (resulting in better mixing).
    _new_stepsize = T.switch(avg_acceptance_rate > target_acceptance_rate,
                              stepsize * stepsize_inc, stepsize * stepsize_dec)
    # maintain stepsize in [stepsize_min, stepsize_max]
    new_stepsize = T.clip(_new_stepsize, stepsize_min, stepsize_max)

    ## ACCEPT RATE UPDATES ##
    # perform exponential moving average
    mean_dtype = theano.scalar.upcast(accept.dtype, avg_acceptance_rate.dtype)
    new_acceptance_rate = T.add(
            avg_acceptance_slowness * avg_acceptance_rate,
            (1.0 - avg_acceptance_slowness) * accept.mean(dtype=mean_dtype))

    return [(positions, new_positions),
            (stepsize, new_stepsize),
            (avg_acceptance_rate, new_acceptance_rate)]



sharedX = lambda X, name: \
        shared(np.asarray(X, dtype=theano.config.floatX), name=name)

class HMC_sampler(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @classmethod
    def new_from_shared_positions(cls, shared_positions, energy_fn,
            initial_stepsize=0.01, target_acceptance_rate=.9, n_steps=20,
            stepsize_dec=0.98,
            stepsize_min=0.001,
            stepsize_max=0.25,
            stepsize_inc=1.02,
 # used in geometric avg. 1.0 would be not moving at all
            avg_acceptance_slowness=0.9,
            seed=12345):

        batchsize = shared_positions.shape[0]

        # allocate shared variables
        stepsize = sharedX(initial_stepsize, 'hmc_stepsize')
        avg_acceptance_rate = sharedX(target_acceptance_rate,
                                      'avg_acceptance_rate')
        s_rng = T.shared_randomstreams.RandomStreams(seed)

        # define graph for an `n_steps` HMC simulation
        accept, final_pos = hmc_move(
                s_rng,
                shared_positions,
                energy_fn,
                stepsize,
                n_steps)

        # define the dictionary of updates, to apply on every `simulate` call
        simulate_updates = hmc_updates(
                shared_positions,
                stepsize,
                avg_acceptance_rate,
                final_pos=final_pos,
                accept=accept,
                stepsize_min=stepsize_min,
                stepsize_max=stepsize_max,
                stepsize_inc=stepsize_inc,
                stepsize_dec=stepsize_dec,
                target_acceptance_rate=target_acceptance_rate,
                avg_acceptance_slowness=avg_acceptance_slowness)

        # compile theano function
        simulate = function([], [], updates=simulate_updates)

        # create HMC_sampler object with the following attributes ...
        return cls(
                positions=shared_positions,
                stepsize=stepsize,
                stepsize_min=stepsize_min,
                stepsize_max=stepsize_max,
                avg_acceptance_rate=avg_acceptance_rate,
                target_acceptance_rate=target_acceptance_rate,
                s_rng=s_rng,
                _updates=simulate_updates,
                simulate=simulate)

    def draw(self, **kwargs):
        """
        Returns a new position obtained after `n_steps` of HMC simulation.
        """
        self.simulate()
        return self.positions.get_value(borrow=False)


def sampler_on_nd_gaussian(sampler_cls, burnin, n_samples, dim=5):
    batchsize=1

    rng = np.random.RandomState(123)
    sigma_prior = .0001
    m_0 = .001
    s_0 = .001
    D= 10
    # Define a covariance and mu for a gaussian
    x  = np.array([1,2,3], dtype=theano.config.floatX)
    y  = np.array([4,8,12], dtype=theano.config.floatX)
    # cov = np.array([[.005,0],[0,.005]], dtype=theano.config.floatX)
    # cov = (cov + cov.T) / 2.
    # cov_inv = np.linalg.inv(cov)

    # Define energy function for a multi-variate Gaussian
    def gaussian_energy(w):
        total_energy = 0
        output_weights = w[3:]
        variance = w[5]
        for i in range(len(x)):
             hidden_layer = T.tanh(w[0]*x[i] +w[1]*x[i] + w[2]*x[i])
             output_layer = (output_weights[0]*hidden_layer+output_weights[1]*hidden_layer )

             total_energy +=  (y[i]-output_layer)**2
        return .5*(m_0 + len(x)*D)*T.log(total_energy+s_0)-T.log(s_0)

    # Declared shared random variable for positions
    position = shared(rng.randn(dim).astype(theano.config.floatX))

    # Create HMC sampler
    sampler = sampler_cls(position, gaussian_energy,
            initial_stepsize=1e-4, stepsize_max=0.5)

    # Start with a burn-in process
    garbage = [sampler.draw() for r in range(burnin)]  #burn-in
    # Draw `n_samples`: result is a 3D tensor of dim [n_samples, batchsize, dim]
    _samples = np.asarray([sampler.draw() for r in range(n_samples)])
    # Flatten to [n_samples * batchsize, dim]
    samples = _samples.T.reshape(dim, -1).T

    #print('****** TARGET VALUES ******')
    # print('target mean:', w)
    # print('target cov:\n', cov)

    print('****** EMPIRICAL MEAN/COV USING HMC ******')
    print('empirical mean: ', samples.mean(axis=0))
    print('empirical_cov:\n', np.cov(samples.T))
    output = samples.mean(axis=0)
    for i in range(len(x)):
        hidden_layer = np.tanh(output[0]*x[i]+output[1]*x[i] + output[2]*x[i])
        output_weights = output[3:]
        output_ = (output_weights[0]*hidden_layer + output_weights[1]*hidden_layer)
        print (output_)
    return sampler

def test_hmc():
    sampler = sampler_on_nd_gaussian(HMC_sampler.new_from_shared_positions,
            burnin=1000, n_samples=10000, dim=5)
    # assert abs(sampler.avg_acceptance_rate.get_value() - sampler.target_acceptance_rate) < .1
    # assert sampler.stepsize.get_value() >= sampler.stepsize_min
    # assert sampler.stepsize.get_value() <= sampler.stepsize_max


test_hmc()