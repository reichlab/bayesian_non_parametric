library('nimble')
library('coda')
Sys.setenv(PATH = paste("/Users/gcgibson/anaconda/bin", Sys.getenv("PATH"), sep=":"))


train_and_predict_ssm <- function(sanJuanDengueData,n_ahead,last_training_point){
  print (n_ahead)
  exec_str <- 'python /Users/gcgibson/Desktop/bayesian_non_parametric/dlm/ssm_pf.py'
  exec_str <- paste(exec_str,n_ahead, sep=" ")
  exec_str <- paste(exec_str,last_training_point,sep=" ")
  dpForecasts <- system(exec_str,intern=TRUE,wait = TRUE)
  dpForecasts <- strsplit(dpForecasts,",")
  dpForecasts <- as.numeric(unlist(dpForecasts))
  dpForecasts
}

