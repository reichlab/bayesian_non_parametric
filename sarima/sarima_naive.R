library('forecast')

train_and_predict_naive_sarima <- function(sanJuanDengueData = sanJuanDengueData,
                                                                  n_ahead = n_ahead, last_training_point = last_training_point){
  
  
preds <- c()  

for (i in seq(last_training_point,(last_training_point+51))){ 
  sarima_model <- auto.arima(sanJuanDengueData[1:(i-n_ahead)])
  sarima_model_fcast <- forecast(sarima_model,h=n_ahead)
  preds <- c(preds, tail(sarima_model_fcast$mean,n=1)[1])
}

return (preds) 

}

