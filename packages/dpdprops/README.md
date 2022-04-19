# dpdprops

A python package for mapping parameters between:
* DPD and fluid properties.
* parameters of coarsed grained models for elastic membranes and macroscopic membrane parameters.

## Installation

	python -m pip install .


## DPD <-> fluid properties

The mapping is computed as explained in the appendix of

	Groot, Robert D., and Patrick B. Warren.
	"Dissipative particle dynamics: Bridging the gap between atomistic and mesoscopic simulation."
	The Journal of chemical physics 107.11 (1997): 4423-4435.

The integrals are adapted to match the modified kernel of the dissipative forces, with the general enveloppe coeficient `s`.

## Membranes

Wrapper classes to hold red blood cell membrane parameters.
Holds default values obtained from UQ studies.
The classes provide a helper to convert from physical quantities to simulation quantities.
