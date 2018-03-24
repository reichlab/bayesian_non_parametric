
Sys.setenv(PATH = paste("/Users/gcgibson/anaconda/bin", Sys.getenv("PATH"), sep=":"))


train_and_predict_dp <- function(sanJuanDengueData,n_ahead,last_training_point,max_lag_val){
    X_train <- cbind(sanJuanDengueData[seq(1,(last_training_point-n_ahead-max_lag_val+1))],sanJuanDengueData[seq(2,(last_training_point-n_ahead-max_lag_val+1+1))],
                    sanJuanDengueData[seq(3,(last_training_point-n_ahead-max_lag_val+1+1+1))],sanJuanDengueData[seq(4,(last_training_point-n_ahead-max_lag_val+3+1))])
    
    X_test <- cbind(
      sanJuanDengueData[seq(last_training_point-max_lag_val-n_ahead+1+1,last_training_point-max_lag_val-n_ahead+1+51+1)],
      sanJuanDengueData[seq(last_training_point-max_lag_val + 1 -n_ahead+1+1,last_training_point-max_lag_val + 1-n_ahead+1+51+1)],
      sanJuanDengueData[seq(last_training_point-max_lag_val + 2-n_ahead+1+1,last_training_point-max_lag_val + 2-n_ahead+1+51+1)],
      sanJuanDengueData[seq(last_training_point-max_lag_val + 3-n_ahead+1+1,last_training_point-max_lag_val + 3-n_ahead+1+51+1)])
    
    y_train <- matrix(sanJuanDengueData[seq(max_lag_val+n_ahead,last_training_point)],ncol=1)
    
    write.table(X_train, file = "/Users/gcgibson/bayesian_non_parametric/X_train.csv", append = FALSE, quote = FALSE, sep = ",",
                eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                col.names = FALSE, qmethod = c("escape", "double"))
    
    write.table(y_train, file = "/Users/gcgibson/bayesian_non_parametric/y_train.csv", append = FALSE, quote = FALSE, sep = ",",
                eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                col.names = FALSE, qmethod = c("escape", "double"))
    
    write.table(X_test, file = "/Users/gcgibson/bayesian_non_parametric/X_test.csv", append = FALSE, quote = FALSE, sep = ",",
                eol = "\n", na = "NA", dec = ".", row.names = FALSE,
                col.names = FALSE, qmethod = c("escape", "double"))
    Sys.sleep(30)
    exec_str <- 'python /Users/gcgibson/bayesian_non_parametric/dp/dp.py'
    dpForecasts <- system(exec_str,intern=TRUE,wait = TRUE)
    dpForecasts <- strsplit(dpForecasts,",")
    dpForecasts <- as.numeric(unlist(dpForecasts))
    dpForecasts

}
