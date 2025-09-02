from pymkm.mktable.core import MKTableParameters, MKTable
from pymkm.io.table_set import StoppingPowerTableSet

import numpy as np

def main():

    ## Select input parameters for specific energy tables generation
    atomic_numbers = [1, 2, 6]
    domain_radius = 0.32 # μm
    nucleus_radius = 3.9 # μm
    beta0 = 0.0615 # 1/Gy^2  
    
    energy = np.logspace(-3, 3, num=100)  # from 1e-3 to 1e-1, inclusive

    path_to_tables = r'C:\Users\giuseppe.magro\Documents\Python Scripts\dEdx_FLUKA2020.0_isotopes\fluka_2020_0_extended'
    sp_table_set = StoppingPowerTableSet.from_directory(directory=path_to_tables)
    sp_table_set.resample_all(energy)

    params = MKTableParameters(
        domain_radius=domain_radius,
        nucleus_radius=nucleus_radius,
        beta0=beta0,
    )

    mk_table = MKTable(parameters=params, sp_table_set=sp_table_set)
    mk_table.compute(ions=atomic_numbers, parallel=True)

    mk_table.plot(ions=atomic_numbers, x="energy", y="z_bar_star_domain", verbose=False)

if __name__ == "__main__":
    main()
