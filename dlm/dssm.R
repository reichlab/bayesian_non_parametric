library(tmvtnorm)
library(ggplot2)
library(Metrics)
SJdat<-read.csv("http://dengueforecasting.noaa.gov/Training/San_Juan_Training_Data.csv")
dat <- SJdat[c("season_week","total_cases")]

n_ahead <- 10

sir <- function(time, state, parameters) {
  
  with(as.list(c(state, parameters)), {
    
    dS <- -beta * S * I
    dI <-  beta * S * I - gamma * I
    dR <-                 gamma * I
    
    return(list(c(dS, dI, dR)))
  })
}


get_point_prediction <- function(n_ahead,last_filtered_state){
  total_prediction <- rep(0,n_ahead)
  for (num_sim in seq(1,10)){
    preds <- c()
    current_state <- last_filtered_state
    for (i in seq(1,n_ahead)){
      init <- c(S = as.numeric(current_state[1]), I = as.numeric(current_state[2]), R = as.numeric(current_state[3]))
      parameters <- c(beta = 1.4, gamma = .1)
      times      <- seq(0, 1, by = 1)
      print (init)
      out <- ode(y = init, times = times, func = sir, parms = parameters,method = "rk4")
      out <- as.data.frame(out)
      out$time <- NULL
      preds <- c(preds,20000*out[1,2])
      current_state <- rdirichlet(1,alpha=c(2e3*out[1,1],2e3*out[1,2],2e3*out[1,3]))
    }
    total_prediction <- total_prediction + preds
  }
  point_prediction_at_n_ahead <- tail(total_prediction,n=1)/10 
  return (point_prediction_at_n_ahead)
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


full_season_ahead_predictions <- c()


model_file = '/Users/gcgibson/Desktop/bayesian_non_parametric/dlm/blob.bug' # BUGS model filename
cat(readLines(model_file), sep = "\n")


for (season_iter in seq(1,20)){
  cases <-dat$total_cases[1:(100+season_iter)]
  t_max = length(cases)
  data = list(t_max=t_max, y = cases,mean_x_init=c(.99,.005,.005))
  sample_data = FALSE # Boolean
  model = biips_model(model_file, data, sample_data=sample_data) # Create Biips model and sample data
  data = model$data()
  n_part = 100 # Number of particles
  variables = c('x','y') # Variables to be monitored
  mn_type = 'fs'; rs_type = 'stratified'; rs_thres = .5 # Optional parameters
  out_smc = biips_smc_samples(model, variables, n_part,
                              type=mn_type, rs_type=rs_type, rs_thres=rs_thres)
  diag_smc = biips_diagnosis(out_smc)
  summ_smc = biips_summary(out_smc, probs=c(.025, .975))
  x_f_mean = summ_smc$x$f$mean
  x_f_quant = summ_smc$x$f$quant
  last_filtered_state <- x_f_mean[,length(cases)]

  full_season_ahead_predictions <- c(full_season_ahead_predictions,get_point_prediction(n_ahead = n_ahead,last_filtered_state = last_filtered_state ))
  
}

print(full_season_ahead_predictions)
