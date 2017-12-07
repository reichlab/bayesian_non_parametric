library(tmvtnorm)
library(ggplot2)
SJdat<-read.csv("http://dengueforecasting.noaa.gov/Training/San_Juan_Training_Data.csv")
dat <- SJdat[c("season_week","total_cases")]


beta <- 5
gamma <- 3.5

predict_ahead <- function(x){
  x[1] <- x[1]+ 1/6*(-beta*x[1]*x[2])
  x[2] <- x[2] + 1/6*(beta*x[1]*x[2]-gamma*x[2])
  x[3] <- x[3] + 1/6*(gamma*x[2])
  return (x)
}



cases <- dat$total_cases[1:100]
require(rbiips)
library(deSolve)
library(MCMCpack)
dMN_dim <- function(s,i,r) {
  # Check dimensions of the input and return dimension of the output of
  # distribution dMN
  3
}
dMN_sample <- function(s,i,r) {
  # Draw a sample of distribution dMN
  sir <- function(time, state, parameters) {
    
    with(as.list(c(state, parameters)), {
      
      dS <- -beta * S * I
      dI <-  beta * S * I - gamma * I
      dR <-                 gamma * I
      
      return(list(c(dS, dI, dR)))
    })
  }
  init <- c(S = s, I = i, R = r)
  parameters <- c(beta = 1.4, gamma = .1)
  times      <- seq(0, 2, by = 1)
  out <- ode(y = init, times = times, func = sir, parms = parameters,method = "rk4")
  out <- as.data.frame(out)
  out$time <- NULL
  return (rdirichlet(1,alpha=c(2e3*out[1,1],2e3*out[1,2],2e3*out[1,3])))
                  
}
biips_add_distribution('dmultinom', 3, dMN_dim, dMN_sample)



model_file = '/Users/gcgibson/Desktop/bayesian_non_parametric/dlm/blob.bug' # BUGS model filename
cat(readLines(model_file), sep = "\n")


t_max = length(cases)
data = list(t_max=t_max, y = cases,mean_x_init=c(.99,.005,.005))

sample_data = FALSE # Boolean
model = biips_model(model_file, data, sample_data=sample_data) # Create Biips model and sample data

data = model$data()

### PMMH
if (FALSE){
  n_burn = 2000 # nb of burn-in/adaptation iterations
  n_iter = 2000 # nb of iterations after burn-in
  thin = 1 # thinning of MCMC outputs
  n_part = 50 # nb of particles for the SMC
  param_names = c('beta','gamma') # name of the variables updated with MCMC (others are updated with SMC)
  latent_names = c('x') # name of the variables updated with SMC and that need to be monitored
  
  
  obj_pmmh = biips_pmmh_init(model, param_names, inits=list(beta=.9,gamma=.1),
                             latent_names=latent_names) # creates a pmmh object
  biips_pmmh_update(obj_pmmh, n_burn, n_part) # adaptation and burn-in iterations
  
  out_pmmh = biips_pmmh_samples(obj_pmmh, n_iter, n_part, thin=thin) # samples
  
  summ_pmmh = biips_summary(out_pmmh, probs=c(.025, .975))
}



### PARTICLE FILTER
n_part = 1000 # Number of particles
variables = c('x','y') # Variables to be monitored
mn_type = 'fs'; rs_type = 'stratified'; rs_thres = .5 # Optional parameters
out_smc = biips_smc_samples(model, variables, n_part,
                            type=mn_type, rs_type=rs_type, rs_thres=rs_thres)
diag_smc = biips_diagnosis(out_smc)
summ_smc = biips_summary(out_smc, probs=c(.025, .975))
x_f_mean = summ_smc$x$f$mean
x_f_quant = summ_smc$x$f$quant
last_state <- x_f_mean[,100]
preds <- c()
for (i in seq(1,52)){
  tmp <- predict_ahead(last_state)
  preds <- c(preds,tmp[2])
  last_state <- tmp
}
print(x_f_mean[2,])
print(cases)
