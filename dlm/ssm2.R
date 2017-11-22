require(rbiips)
library(MCMCpack)
dMN_dim <- function(s,i,r) {
  # Check dimensions of the input and return dimension of the output of
  # distribution dMN
  3
}
dMN_sample <- function(s,i,r) {
  # Draw a sample of distribution dMN

  rsamp <- rdirichlet(1,c(1e4*s,1e4*i,1e4*r))
  c(rsamp[1],rsamp[2],rsamp[3])
}
biips_add_distribution('ddirch', 3, dMN_dim, dMN_sample)



model_file = '/Users/gcgibson/Desktop/bayesian_non_parametric/dlm/blob.bug' # BUGS model filename
cat(readLines(model_file), sep = "\n")

par(bty='l')
light_blue = rgb(.7, .7, 1)
light_red = rgb(1, .7, .7)

t_max = 20
mean_x_init = 1
prec_x_init = 1/5
prec_x = 1/10
true_data <- rep(10,t_max)
log_prec_y_true = log(1) # True value used to sample the data
data = list(t_max=t_max, y = true_data,prec_x_init=prec_x_init,
            prec_x=prec_x, 
            mean_x_init=c(1000.0,10.0,10.0))

sample_data = FALSE # Boolean
model = biips_model(model_file, data, sample_data=sample_data) # Create Biips model and sample data

data = model$data()
n_part = 10000 # Number of particles
variables = c('x','y') # Variables to be monitored
mn_type = 'fs'; rs_type = 'stratified'; rs_thres = 0.5 # Optional parameters



out_smc = biips_smc_samples(model, variables, n_part,
                            type=mn_type, rs_type=rs_type, rs_thres=rs_thres)
diag_smc = biips_diagnosis(out_smc)
summ_smc = biips_summary(out_smc, probs=c(.025, .975))


x_f_mean = summ_smc$x$f$mean
x_f_quant = summ_smc$x$f$quant


filtered_states = 2000*x_f_mean[2,]
