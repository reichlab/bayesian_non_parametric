using Mamba

## Define a new unknown Distribution type.
## The definition must be placed within an unevaluated quote block; however, it
## may be easiest to first develop and test the definition apart from the block.

extensions = quote

  ## Type definition and constructor
  type UnknownDist <: ContinuousUnivariateDistribution
    mu::Float64
    sigma::Float64
    UnknownDist(mu, sigma) = new(float64(mu), float64(sigma))
  end

  ## Method functions minimum, maximum, insupport, and logpdf are required

  ## The minimum and maximum support values
  minimum(d::UnknownDist) = -Inf
  maximum(d::UnknownDist) = Inf

  ## A logical indicating whether x is in the support
  insupport(d::UnknownDist, x::Real) = true

  ## The normalized or unnormalized log-density value
  function logpdf(d::UnknownDist, x::Real)
    -log(d.sigma) - 0.5 * ((x - d.mu) / d.sigma)^2
  end

  ## Make the type available outside of Mamba (optional)
  export UnknownDist

end

## Add the new type to Mamba
eval(Mamba, extensions)

## If exported, test its constructor and functions here (optional)
d = UnknownDist(1.0, 2.0)
minimum(d)
maximum(d)
insupport(d, 1.0)
logpdf(d, 1.0)

## Implement a Mamba model using the unknown distribution
model = Model(

  y = Stochastic(1,
    @modelexpr(mu, s2,
      begin
        sigma = sqrt(s2)
        Distribution[
          UnknownDist(mu[i], sigma)
          for i in 1:length(mu)
        ]
      end
    ),
    false
  ),

  mu = Logical(1,
    @modelexpr(xmat, beta,
      xmat * beta
    ),
    false
  ),

  beta = Stochastic(1,
    :(MvNormal(2, sqrt(1000)))
  ),

  s2 = Stochastic(
    :(InverseGamma(0.001, 0.001))
  )

)

## Sampling Scheme
scheme = [NUTS([:beta]),
          Slice([:s2], [3.0])]

## Sampling Scheme Assignment
setsamplers!(model, scheme)

## Data
line = (Symbol => Any)[
  :x => [1, 2, 3, 4, 5],
  :y => [1, 3, 3, 3, 5]
]
line[:xmat] = [ones(5) line[:x]]

## Initial Values
inits = Dict{Symbol,Any}[
  [:y => line[:y],
   :beta => rand(Normal(0, 1), 2),
   :s2 => rand(Gamma(1, 1))]
  for i in 1:3]

## MCMC Simulation
sim = mcmc(model, line, inits, 10000, burnin=250, thin=2, chains=3)
describe(sim)
