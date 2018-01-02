
 model SIR {
   const N = 1000
   state S, I, R
   obs y
   
   sub initial {
     S <- N - 1
     I <- 1
     R <- 0
   }
   sub transition {
     inline i_beta = 2  
     inline i_gamma = 1.4 
     ode (alg='RK4(3)', h=1e-1, atoler=1e-2, rtoler=1e-5) {
       dS/dt = - i_beta * S * I / N
       dI/dt = i_beta * S * I / N - i_gamma * I
       dR/dt = i_gamma * I
     }
   }
   sub observation {
     y ~ gaussian(mean=I,std=1000)
   }
 }
