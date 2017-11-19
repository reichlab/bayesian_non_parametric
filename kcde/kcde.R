library(dlm)
library(ggplot2)
library(forecast)
library(np)
require(rnn)
library(quantmod)
library(RSNNS)
library(pomp)
library(bssm)
library('nimble')
source('/Users/gcgibson/Desktop/bayesian_non_parametric/dengue_sj_experiments/sarima')

### SET END OF TRAINING / START OF TESTING

train_and_predict_kcde <- function(sanJuanDengueData,n_ahead,last_training_point,max_lag_val){

### UTILITY METHOD TO CREAT AR(4) TRAINING DATA
lagged_4 <- cbind(sanJuanDengueData[seq(1,(last_training_point-n_ahead-max_lag_val+1))],sanJuanDengueData[seq(2,(last_training_point-n_ahead-max_lag_val+1+1))],
                  sanJuanDengueData[seq(3,(last_training_point-n_ahead-max_lag_val+1+1+1))],sanJuanDengueData[seq(4,(last_training_point-n_ahead-max_lag_val+3+1))])


### FIT KCDE
bw <- npcdensbw(
  xdat = lagged_4,
  ydat = matrix(sanJuanDengueData[seq(max_lag_val+n_ahead,last_training_point)],ncol=1),
  nmulti = 1,
  remin = FALSE,
  bwtype = "adaptive_nn",
  bwmethod = "cv.ml")

get_discrete_prob_from_np <- function(X, y, bw) {
  X # so not an empty promise
  bw # so not an empty promise
  fitted(npcdens(
    exdat = X,
    eydat = y,
    bws = bw))
}

val_grid <- seq(from = -1, to = 100, by = 1)

X_test <- cbind(
  sanJuanDengueData[seq(last_training_point-max_lag_val-n_ahead+1+1,last_training_point-max_lag_val-n_ahead+1+51+1)],
  sanJuanDengueData[seq(last_training_point-max_lag_val + 1 -n_ahead+1+1,last_training_point-max_lag_val + 1-n_ahead+1+51+1)],
  sanJuanDengueData[seq(last_training_point-max_lag_val + 2-n_ahead+1+1,last_training_point-max_lag_val + 2-n_ahead+1+51+1)],
  sanJuanDengueData[seq(last_training_point-max_lag_val + 3-n_ahead+1+1,last_training_point-max_lag_val + 3-n_ahead+1+51+1)])

kcdeForecasts <- c()
for (i in seq(1,nrow(X_test))){
  mean_response <- 0
  for(response_val in val_grid) {
    mean_response <- mean_response +  response_val*get_discrete_prob_from_np(
      X = matrix(X_test[i],ncol=4),
      y = response_val,
      bw = bw)
  }
  kcdeForecasts <- c(kcdeForecasts,mean_response)
}
kcdeForecasts
}





