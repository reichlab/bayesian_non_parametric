import numpy as np
import pymc3 as pm
import theano.tensor as tt

if __name__ == "__main__":
    # Generate some data.
    X = np.random.randn(100, 4)
    X[:, 0] = 1. # Intercept column.
    Y = np.dot(X, [0.5, 0.1, 0.25, 1.]) + 0.1 * np.random.randn(X.shape[0])
    
    # Pymc3 model.
    model = pm.Model()
    with model:
        # Define beta priors.
        B = pm.Normal("B", mu=0.0, sd=1.0, shape=X.shape[1])
        # Model variables.
        Ymu = tt.dot(X, B.T)
        Ysd = pm.Uniform("Ysd", 0., 10.)
        # Data likelihood.
        required_argument = 123.45
        logp = -tt.log(Ysd * tt.sqrt(2.0 * np.pi)) - (tt.sqrt(Y - Ymu) / (2.0 * Ysd * Ysd))
        def logp_func(required_argument):
            return logp.sum()
        logpvar = pm.DensityDist("logpvar", logp_func, observed={"required_argument": required_argument})
        # Sample.
        start = pm.find_MAP(model=model)
        step = pm.NUTS(scaling=start)
        trace = pm.sample(100, start=start, step=step)
    print(pm.summary(trace))
