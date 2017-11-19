
model <- ng_bsm(sanJuanDengueDataTrainTS, sd_level = halfnormal(0.1, 1),
                sd_slope=halfnormal(0.01, 0.1), distribution = "poisson")
mcmc_poisson <- run_mcmc(model, n_iter = 5000, nsim = 10)
future_model <- model
future_model$y <- ts(rep(NA, 10), start = end(model$y),
                     frequency = frequency(model$y))
pred <- predict(mcmc_poisson, future_model,
                probs = seq(0.05,0.95, by = 0.05))
print (pred$mean)

#### POISSON SSM
stateSpaceCode <- nimbleCode({
  a ~ dunif(-0.9999, 0.9999)
  b ~ dnorm(0, sd = 1000)
  sigPN ~ dunif(1e-04, 1)
  sigOE ~ dunif(1e-04, 1)
  
  x[1] ~ dnorm(b/(1 - a), sd = sigPN/sqrt((1-a*a)))
  y[1] ~ dpois(lambda = exp(x[1]))
  
  for (i in 2:t) {
    x[i] ~ dnorm(a * x[i - 1] + b, sd = sigPN)
    y[i] ~ dpois(lambda = exp(x[i]))
  }
})
