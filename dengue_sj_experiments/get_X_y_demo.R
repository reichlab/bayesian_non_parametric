## Demo of using compute_offset_obs_vecs function in utils.R to get
## X_train, y_train, X_test

library(dplyr)
library(readr)

## Assumes working directory is bayesian_non_parametric root directory
source("dengue_sj_experiments/utils.R")

# perhaps confusingly, lag indexing is 0-based here, so max_lag of 3 returns 4
# columns of data for X: relative to time t, (y_{t-3}, y_{t-2}, y_{t-1}, y_{t})
max_lag <- 3
# relative to time t, predict time t + 10
prediction_horizon <- 10

## Train: data from 1990/1991 through 2004/2005 seasons
temp <- get_X_train_y_train(
  max_lag = max_lag,
  prediction_horizon = prediction_horizon)

X_train <- temp$X
y_train <- temp$Y


## Eval: data from 2005/2006 through 2008/2009 seasons
temp <- get_X_eval(
  max_lag = max_lag,
  prediction_horizon = prediction_horizon)

X_eval <- temp$X_eval
