library("GPfit")

# Casey's original code:
x <-seq(1,10,.1)
y <- sin(x)

xvec <-seq(10,20,.1)
GPmodel = GP_fit(x,y);
GPprediction = predict.GP(GPmodel,xvec);
plot(GPprediction$Y_hat)
points(sin(xvec),col='red')




# "auto-regressive" specification with short-term lags -- made training data a little longer too, to get the more lags in
# max_lag 1 fails
x <-seq(1,30,.1)
y <- sin(x)

X <- NULL
max_lag <- 1
for(lag_val in seq_len(max_lag)) {
  #  X <- cbind(lag(y, lag_val), X)
  X <- cbind(lag((y + 1)/2, lag_val), X)
}
non_na_inds <- apply(X, 1, function(Xrow) {!any(is.na(Xrow))})
X <- X[non_na_inds, ]
y <- y[non_na_inds]

GPmodel = GP_fit(X,(y + 1)/2)



xvec <-seq(30 - max_lag * 0.1, 30, .1)
ytest <- sin(xvec)

Xtest <- NULL
for(lag_val in seq_len(max_lag)) {
  #  Xtest <- cbind(lag(y, lag_val), Xtest)
  Xtest <- cbind(lag((ytest + 1)/2, lag_val), Xtest)
}
non_na_inds <- apply(Xtest, 1, function(Xrow) {!any(is.na(Xrow))})
Xtest <- Xtest[non_na_inds, , drop = FALSE]


# predict recursively
pred <- (ytest + 1)/2
num_steps_pred <- 60
for(i in seq_len(num_steps_pred)) {
  Xtest <- cbind(Xtest, tail(pred, 1))[, -1, drop = FALSE]
  pred <- c(pred, predict.GP(GPmodel,Xtest)$Y_hat)
}
plot(pred * 2 - 1)

xvec <-seq(30 - max_lag * 0.1, 30 + num_steps_pred, .1)
points(sin(xvec),col='red')



# "auto-regressive" specification with short-term lags -- made training data a little longer too, to get the more lags in
# max_lag 4 works
x <-seq(1,30,.1)
y <- sin(x)

X <- NULL
max_lag <- 4
for(lag_val in seq_len(max_lag)) {
  #  X <- cbind(lag(y, lag_val), X)
  X <- cbind(c(rep(NA, lag_val), (y[seq_len(length(y) - lag_val)] + 1)/2), X)
}
non_na_inds <- apply(X, 1, function(Xrow) {!any(is.na(Xrow))})
X <- X[non_na_inds, ]
y <- y[non_na_inds]

GPmodel = GP_fit(X,(y + 1)/2)



xvec <-seq(30 - max_lag * 0.1, 30, .1)
ytest <- sin(xvec)

Xtest <- NULL
for(lag_val in seq_len(max_lag)) {
  #  Xtest <- cbind(lag(y, lag_val), Xtest)
  Xtest <- cbind(c(rep(NA, lag_val), (ytest[seq_len(length(ytest) - lag_val)] + 1)/2), Xtest)
#  Xtest <- cbind(lag((ytest + 1)/2, lag_val), Xtest)
}
non_na_inds <- apply(Xtest, 1, function(Xrow) {!any(is.na(Xrow))})
Xtest <- Xtest[non_na_inds, , drop = FALSE]


# predict recursively
pred <- (ytest + 1)/2
num_steps_pred <- 60
for(i in seq_len(num_steps_pred)) {
  Xtest <- cbind(Xtest, tail(pred, 1))[, -1, drop = FALSE]
  pred <- c(pred, predict.GP(GPmodel,Xtest)$Y_hat)
}
plot(pred * 2 - 1)

xvec <-seq(30 - max_lag * 0.1, 30 + num_steps_pred, .1)
points(sin(xvec),col='red')
