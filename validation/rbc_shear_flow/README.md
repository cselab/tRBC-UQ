# rbc shear flow

A single RBC under shear flow.

## Important note on the choice of units:

We only match the Capillary number and the cell dimensionless numbers (FvK...).
Re and Ma are not matched for computational reasons.
For that reason, we do not have a clear time scale and mass scale, those are arbitrary here.
In principle, one could set the "real" time scale but because of the larger Re we would have a "wrong" mass scale.
For that reason we decide here to leave the time scale arbitrary, the user should report the output of the simulation in dimensionless units when it comes to time or mass, e.g. TTF / shear_rate.
