library(dlm)

train_and_predict_dlm <- function(sanJuanDengueData,n_ahead,last_training_point){
    dlmForecasts <- c()
    sanJuanDengueDataTrainTS <- ts(sanJuanDengueData[seq(1,last_training_point)], start = 1, frequency = 52)
    dengue_dlm <- function(parm){
      return(dlmModTrig(s=52,dW = exp(parm[4]))+ dlmModARMA(ar =parm[1],ma=parm[2],sigma2=exp(parm[3])))
    }
    fit1 <- dlmMLE(y=sanJuanDengueDataTrainTS,parm=c(1,1,1,1),build=dengue_dlm,hessian=T)
    coef <- fit1$par
    
    mod <- dlmModARMA(ar =coef[1],ma=coef[2],sigma2=exp(coef[3])) +dlmModTrig(s=52,dW = exp(coef[4]))
    
    for (i in seq(last_training_point+1-n_ahead,last_training_point+1+51-n_ahead)){
      outF <- dlmFilter(sanJuanDengueData[seq(1,i)], mod)
      dlmfore <- dlmForecast(outF, nAhead = n_ahead)
      dlmForecasts <- c(dlmForecasts,tail(dlmfore$f,n=1))
    }
    
    
dlmForecasts
}