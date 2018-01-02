library('nimble')

stateSpaceCode <- nimbleCode({
  a ~ dunif(-0.9999, 0.9999)
  b ~ dnorm(0, sd = 1000)
  sigPN ~ dunif(1e-04, 1)
  sigOE ~ dunif(1e-04, 1)
  
  x[1] ~ dnorm(b/(1 - a), sd = sigPN/sqrt((1-a*a)))
  y[1] ~ dpois(lambda = x[1])
  
  for (i in 2:t) {
    x[i] ~ dnorm(a * x[i - 1] + b, sd = sigPN)
    y[i] ~ dpois(lambda = x[i])
  }
})

data <- list(
  y = c(rep(100,16))
)
constants <- list(
  t = 16
)
inits <- list(
  a = 0,
  b = .5,
  sigPN = .1,
  sigOE = .05
)

## build the model
stateSpaceModel <- nimbleModel(stateSpaceCode,
                               data = data,
                               constants = constants,
                               inits = inits,
                               check = FALSE)


bootstrapFilter <- buildBootstrapFilter(stateSpaceModel, nodes = 'x')
compiledList <- compileNimble(stateSpaceModel, bootstrapFilter)
compiledList$bootstrapFilter$run(10000)
posteriorSamples <- as.matrix(compiledList$bootstrapFilter$mvEWSamples)
hist(posteriorSamples)
mean(posteriorSamples)
dpois(lambda=mean(posteriorSamples),n=100)


# 
# 
# stateSpaceMCMCconf <- configureMCMC(stateSpaceModel, nodes = NULL)
# 
# ## add a block pMCMC sampler for a, b, sigPN, and sigOE 
# stateSpaceMCMCconf$addSampler(target = c('a', 'b', 'sigPN', 'sigOE'),
#                               type = 'RW_PF_block', control = list(latents = 'x'))
# 
# ## build and compile pMCMC sampler
# stateSpaceMCMC <- buildMCMC(stateSpaceMCMCconf)
# compiledList <- compileNimble(stateSpaceModel, stateSpaceMCMC, resetFunctions = TRUE)
# 
# 
# compiledList$stateSpaceMCMC$run(5000)
# library('coda')
# 
# posteriorSamps <- as.mcmc(as.matrix(compiledList$stateSpaceMCMC$mvSamples))
# print (posteriorSamps[,'a'])
# print (posteriorSamps[,'b'])