require(ForecastFramework)
require(R6)
require(forecast)

#To install rbiips: 
#install_dir <- file.path(tempdir(), "rbiips")
#system(paste("git clone --recursive", shQuote("https://github.com/biips/rbiips.git"), shQuote(install_dir)))
#devtools::install(install_dir)

require(rbiips)
library(deSolve)
library(MCMCpack)

get_point_prediction <- function(n_ahead,last_filtered_state){
  total_prediction <- rep(0,n_ahead)
  for (num_sim in seq(1,100)){
    preds <- c()
    current_state <- last_filtered_state
    for (i in seq(1,n_ahead)){
      init <- c(S = as.numeric(current_state[1]), I = as.numeric(current_state[2]), R = as.numeric(current_state[3]))
      parameters <- c(beta = 1.4, gamma = .1)
      times      <- seq(0, 1, by = 1)
      out <- ode(y = init, times = times, func = sir, parms = parameters,method = "rk4")
      out <- as.data.frame(out)
      out$time <- NULL
      preds <- c(preds,20000*out[1,2])
      current_state <- rdirichlet(1,alpha=c(2e3*out[1,1],2e3*out[1,2],2e3*out[1,3]))
    }
    total_prediction <- total_prediction + preds
  }
  point_prediction_at_n_ahead <- tail(total_prediction,n=1)/100 
  return (point_prediction_at_n_ahead)
}


create_dssm_model <- function(data){
  sir <- function(time, state, parameters) {
    with(as.list(c(state, parameters)), {
      dS <- -beta * S * I
      dI <-  beta * S * I - gamma * I
      dR <-                 gamma * I
      
      return(list(c(dS, dI, dR)))
    })
  }
  
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
  
  t_max = length(cases)
  data = list(t_max=t_max, y = cases,mean_x_init=c(.99,.005,.005))
  sample_data = FALSE # Boolean
  model = biips_model(model_file, data, sample_data=sample_data)
  n_part = 1000 # Number of particles
  variables = c('x','y') # Variables to be monitored
  mn_type = 'fs'; rs_type = 'stratified'; rs_thres = .5 # Optional parameters
  out_smc = biips_smc_samples(model, variables, n_part,
                              type=mn_type, rs_type=rs_type, rs_thres=rs_thres)
  diag_smc = biips_diagnosis(out_smc)
  summ_smc = biips_summary(out_smc, probs=c(.025, .975))
  return (summ_smc$x$f$mean)
  
  
}




SIRSSMModel <- R6Class(
  inherit = ForecastModel,
  private = list(
    .data = NULL,        ## every model should have this
    .models = list(),    ## specific to models that are fit separately for each location
    .nsim = 100,         ## models that are simulating forecasts need this
    .lambda = list(),    ## specific to ARIMA models
    .period = integer(0) ## specific to SARIMA models
  ),
  public = list(
    ## data will be MatrixData
    fit = function(data) {
      if("fit" %in% private$.debug){browser()}
      ## stores data for easy access and checks to make sure it's the right class
      private$.data <- IncidenceMatrix$new(data)
      
      ## for each location/row
      for (row_idx in 1:private$.data$nrow) {
        ### need to create a y vector with incidence at time t
        y <- private$.data$subset(rows = row_idx, mutate = FALSE)
        
        ## private$.models[[row_idx]] <- something
         private$.models[[row_idx]] <- create_dssm_model(y$mat)
                                                 
      }
      
    },
    forecast = function(newdata = private$.data, steps) {
      if("forecast" %in% private$.debug){browser()}
      
      nmodels <- length(private$.models)
      # sim_forecasts <- SimulatedIncidenceMatrix$new(array(dim = c(nmodels, steps, private$.nsim)))
      sim_forecasts <- array(dim = c(nmodels, steps, private$.nsim))
      
      for(sim in 1:private$.nsim) {
        for(model_idx in 1:length(private$.models)) {
          tmp_dssm <- get_point_prediction(steps,private$.models[[model_idx]])
          # sim_forecasts$mutate(rows=model_idx, sims=sim, data=tmp_arima)
          sim_forecasts[model_idx, , sim] <- tmp_dssm
        }
      }
      rc <- SimulatedIncidenceMatrix$new(sim_forecasts)
      return(IncidenceForecast$new(rc, forecastTimes = rep(TRUE, steps)))
    },
    initialize = function(period, nsim=100) { 
      ## this code is run during SARIMAModel$new()
      ## need to store these arguments within the model object
      private$.nsim <- nsim
      private$.period <- period
    },
    predict = function(newdata) {
      stop("predict method has not been written.")
    }
  ),
  active = list(
    ## This list determines how you can access pieces of your model object
    data = function(value) {
      ## use this form when you want this parameter to be un-modifiable
      if(!missing(value))
        stop("Writing directly to the data is not allowed.")
      return(private$.data)
    },
    models = function(value) {
      ## use this form when you want this parameter to be un-modifiable
      if(!missing(value))
        stop("Writing directly to the models is not allowed.")
      return(private$.models)
    },
    nsim = function(value) {
      ## use this form when you want to be able to change this parameter
      private$defaultActive(type="private", "nsim", val=value)
    },
    lambda = function(value) {
      ## use this form when you want this parameter to be un-modifiable
      if(!missing(value))
        stop("Writing directly to lambda is not allowed.")
      return(private$.lambda)
    },
    period = function(value) {
      ## use this form when you want this parameter to be un-modifiable
      if(!missing(value))
        stop("Writing directly to the model period is not allowed.")
      return(private$.period)
    }
  )
)
