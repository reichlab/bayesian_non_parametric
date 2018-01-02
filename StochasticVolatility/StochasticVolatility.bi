/**
 * Stochastic volatility model for S&P 500 log return data.
 */
model StochasticVolatility {
  noise eta;
  state S;
  state I;
  state R;
  obs y;


  sub initial {


    S ~ Binomial(200,.9);
    I ~ Binomial(200-I,.05);
    R ~ Binomial(200- I-R,.05);
      
}

  sub transition {
    eta ~ normal();
    S <- S + 1/6(-2*S*I) +eta
    I <- I + 1/6(2*S*I - 1.4*I) + eta
    R <- 1.4*I + eta
  }

  sub observation {
    y ~ normal(I,.1);
  }
}
