# Bayesian inference of the LWM + Juelicher  RBC model

## Experiments:

* equilibrium shape
* stretching
* (relaxation)

## Procedure:

For a single set of parameters, the experiments are conveniently ordered as follows:

1. create the stress-free shape with the desired reduced volume
2. run equilibration of the cell
3. run stretching from the equilibrium shape
4. (run relaxation from the stretched shape)

## Surrogate:

We build a surrogate to map the variables (v, mu, ka, kb, b2, Fext (, eta_m)) to the output of each experiment (D, hmin, hmax) for equilibrium, (D0, D1) for stretching (, (tc) for relaxation).
