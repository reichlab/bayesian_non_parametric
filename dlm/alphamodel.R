library(dlm)

alpha_1 <- 80
alpha_2 <- 1

X <- c(1,alpha_1,alpha_1,1,1,alpha_2,alpha_2,1,1)

JFF <- 1
FF <-1 
GG <- 1

dlmMod <- dlm(m0=1,C0=1,FF=FF,GG=GG,W=1,V =1,X=X,JFF=JFF)

yt <- sin(1:10)

filter_results <- dlmFilter(yt,dlmMod)

plot(filter_results$f)