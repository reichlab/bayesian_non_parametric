library('np')
library('dplyr')
library('ggplot2')

f<-c(4, 5, 4, 3, 6, 2, 4, 5, 10, 6, 8, 2, 6, 17, 23, 13, 21, 28, 24, 20, 40, 27, 42, 33, 43, 37, 57, 71, 44, 56, 53, 52, 47, 26, 27, 21, 21, 26, 34, 37, 17, 19, 25, 18, 21, 17, 17, 16, 16, 15, 23, 16, 17, 12, 17, 10, 15, 19, 21, 14)

min_val <- 0
max_val <- 200

X_train <- f[1:49]
y_train <- f[2:50]
X_test <- f[50:59]
y_test <- f[51:60]

# to evaluate at a grid of all X in 1 to 60:
# X_test <- 1:60

bw <- npcdensbw(
  xdat = X_train,
  ydat = y_train,
  nmulti = 2,
  remin = FALSE,
  bwtype = "adaptive_nn",
  bwmethod = "cv.ml")


get_discrete_prob_from_np <- function(X, y, bw) {
  X # so not an empty promise
  bw # so not an empty promise

  npcdens_wrapper <- function(y) {
    fitted(npcdens(
      exdat = rep(X, length(y)),
      eydat = y,
      bws = bw))
  }

  integrate(npcdens_wrapper, lower = y - 0.5, upper = y + 0.5)$value
}

val_grid <- seq(from = min_val, to = max_val, by = 1)
full_prediction <- matrix(NA,
  nrow = length(X_test),
  ncol = max_val - min_val + 1) %>%
 as.data.frame() %>%
 `colnames<-`(val_grid) %>%
 mutate(
   max_val = NA,
   mean_val = NA
 )

for(i in seq_along(X_test)) {
 for(response_val in val_grid) {
   full_prediction[i, as.character(response_val)] <- get_discrete_prob_from_np(
     X = X_test[i],
     y = response_val,
     bw = bw)
 }
 full_prediction$max_val[i] <- which.max(full_prediction[i, as.character(val_grid)])
 full_prediction$mean_val[i] <- weighted.mean(
   x = val_grid,
   w = full_prediction[i, as.character(val_grid)])
}

mean((full_prediction$max_val - y_test)^2)


res <- data.frame(
  X = X_test,
  true = y_test,
  np = full_prediction$mean_val,
  bnn = c(16.114642329667483, 25.63453717767942, 17.453254035600153, 18.752848755983457, 11.927155716552617, 18.752848755983457, 9.126103980527375, 16.114642329667483, 21.22423127592392, 23.51799800174392),
  bnn2 = c(16.56270, 24.09805, 17.49597, 18.43366, 13.79864, 18.43366, 11.99446, 16.56270, 20.31844, 22.20921),
  brt = c(17.602343, 19.769426, 17.502394, 18.841502, 12.751916, 18.841502,  8.125309, 17.602343, 24.517237, 23.975239),
  svm = c(17.35732, 21.05650, 17.81676, 18.27723, 15.98617, 18.27723, 15.07896, 17.35732, 19.20090, 20.12758)
)

mean((res$true - res$np)^2)
mean((res$true - res$bnn)^2)
mean((res$true - res$bnn2)^2)
mean((res$true - res$brt)^2)
mean((res$true - res$svm)^2)

ggplot() +
  geom_point(aes(x = X, y = y),
    data = data.frame(X = X_train, y = y_train)) +
  geom_point(aes(x = X, y = prediction, color = model),
    data = res %>%
      gather_("model", "prediction", c("true", "np", "bnn", "bnn2", "brt", "svm"))) +
  geom_abline(intercept = 0, slope = 1) +
  theme_bw()
