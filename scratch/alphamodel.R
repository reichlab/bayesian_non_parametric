library(dlm)
SJdat<-read.csv("http://dengueforecasting.noaa.gov/Training/San_Juan_Training_Data.csv")
dat <- SJdat[c("season_week","total_cases")]
a<-c()
t<-c()

for (i in 1:19) {
  a[i] <- max(dat[((i-1)*52+1):(i*52),2])
  t[i] <- which.max(dat[((i-1)*52+1):(i*52),2])
}

X <- c()
for (i in 1:19){
   X <-c(X,c(rep(1,22),rep(a[i],13),rep(1,17)))
}
#X <- rep(c(rep(1,22),rnorm(13,median(a),1),rep(1,17)),19)
X <- cbind(X,rep(0,988))


JFF <- matrix(c(1,0,0,0),1,4)
FF <- matrix(c(1,0,1,0),1,4)

GG <- matrix(rep(0,16),nrow=4,ncol=4)
GG[1,1] = 1
GG[3,3] = cos(2*pi/52)
GG[3,4] = sin(2*pi/52)
GG[4,3] = -sin(2*pi/52)
GG[4,4] = cos(2*pi/52)


build1 <- function(x){
  
  dlmMod <- dlm(m0=rep(0,4),C0=1e+07*diag(4),FF=FF,GG=GG,W=diag(c(exp(x[1]),0,0,0)),
                V =exp(x[2]),
                X=X, JFF=JFF)
}


mle <- dlmMLE(dat$total_cases,parm = c(0,0),build = build1)



dlmMod <- dlm(m0=rep(0,4),C0=1e+07*diag(4),FF=FF,GG=GG,W=diag(c(exp(mle$par[1]),0,0,0)),
              V =exp(mle$par[2]),
              X=X, JFF=JFF)




filter_results <- dlmFilter(dat$total_cases,dlmMod)
plot.ts(dat$total_cases, ylab="Incidence") # filtered
lines(dropFirst(filter_results$m[,1]+filter_results$m[,3]),col="red") # original data
legend("topright", legend=c("filtered","data"), col=c("red","black"), lty=1)

residuals_ <- residuals(filter_results)$res
qqnorm(residuals_)
qqline(residuals_)

