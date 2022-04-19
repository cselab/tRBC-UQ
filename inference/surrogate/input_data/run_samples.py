#!/usr/bin/env python

import numpy as np
import os
import pandas as pd
import pint
import sys

from mpi4py import MPI

from dpdprops import (JuelicherLimRBCDefaultParams,
                      equivalent_sphere_radius)

here = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(here, "..", ".."))

from model import (run_experiments,
                   ExperimentsOutput)

from prior import surrogate_variables_dict

def get_RA_(ureg):
    """
    Return the radius of a sphere with same area as a RBC in units of the given pint.UnitRegistry.
    """
    params = JuelicherLimRBCDefaultParams(ureg)
    return equivalent_sphere_radius(area=params.A0)

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Run all experiments for multiple samples in parameters space')
    parser.add_argument('--sample-list', type=str, required=True, help="csv file that contains the samples renormalized in the [0, 1] interval.")
    parser.add_argument('--mesh-sphere', type=str, required=True, help="Initial spherical mesh.")
    parser.add_argument('--mesh-ini-eq', type=str, required=True, help="Initial mesh to use for equilibration step.")
    parser.add_argument('--out', type=str, default="samples.csv", help="output csv that contains parameters and output of the model.")
    args = parser.parse_args(argv)

    comm = MPI.COMM_WORLD
    ureg = pint.UnitRegistry()

    RA_ = get_RA_(ureg)

    # hack for developping on on barry
    if os.uname()[1] == 'barry.ethz.ch':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(comm.rank % 4)

    df = pd.read_csv(args.sample_list)

    ntotal = len(df['v'])

    nlocal = (ntotal + comm.size - 1) // comm.size

    start = nlocal * comm.rank
    end   = nlocal * (comm.rank + 1)
    end   = min([ntotal, end])
    nlocal = end - start

    subcomm = comm.Split(color=comm.rank, key=comm.rank)

    data = []

    for i in range(start, end):
        def get_scaled(name: str):
            var = surrogate_variables_dict[name]
            return var.low() + (var.high() - var.low()) * df[name][i]

        v    = get_scaled("v")
        mu   = get_scaled("mu")
        FvK  = get_scaled("FvK")
        b2   = get_scaled("b2")
        etam = get_scaled("etam")
        Fext = get_scaled("Fext")

        p = [v, mu, FvK, b2, etam, Fext]

        mu *= ureg.uN / ureg.m
        ka = mu
        kb = (mu * RA_**2 / FvK).to(ureg.J)
        etam *= ureg.Pa * ureg.s * ureg.um
        Fext *= ureg.pN

        # print(f"v={v}, mu={mu}, ka={ka}, kb={kb} (FvK={FvK}), b2={b2}, etam={etam} Fext={Fext}")

        rbc_params = JuelicherLimRBCDefaultParams(ureg,
                                                  mu=mu,
                                                  ka=ka,
                                                  kappab=kb,
                                                  b2=b2,
                                                  eta_m=etam)

        try:
            is_master, res = run_experiments(ureg=ureg,
                                             reduced_volume=v,
                                             rbc_params=rbc_params,
                                             Fext_=[Fext],
                                             mesh_sphere=args.mesh_sphere,
                                             mesh_ini_eq=args.mesh_ini_eq,
                                             comm_address=MPI._addressof(subcomm),
                                             run_stretch=True,
                                             run_relax=True,
                                             verbose=False)

            qoi = [res.eq_D.to(ureg.um).magnitude,
                   res.eq_hmin.to(ureg.um).magnitude,
                   res.eq_hmax.to(ureg.um).magnitude,
                   res.stretch_D0[0].to(ureg.um).magnitude,
                   res.stretch_D1[0].to(ureg.um).magnitude,
                   res.relax_tc.to(ureg.s).magnitude]

            data.append((p + qoi))
            print(*(p + qoi))
            sys.stdout.flush()

        except:
            print(f"A problem occured with parameters {p}. Skipped.")
            sys.stdout.flush()


    data = np.array(data)
    data = comm.gather(data, root=0)

    if comm.rank == 0:
        data = np.concatenate(data)

        keys = ['v', 'mu', 'FvK', 'b2', 'etam', 'Fext',
                'eq_D', 'eq_hmin', 'eq_hmax', 'stretch_D0', 'stretch_D1', 'relax_tc']

        df = pd.DataFrame({key: data[:,i] for i, key in enumerate(keys)})
        df.to_csv(args.out, index=False)




if __name__ == '__main__':
    main(sys.argv[1:])
