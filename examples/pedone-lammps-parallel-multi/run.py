import ezff
from ezff.interfaces import gulp, vasp, lammps
import numpy as np

# DEFINE GROUND TRUTHS
gt_bulk_modulus = 37 #GPa
gt_structure = vasp.read_atomic_structure('ground_truths/POSCAR')

def my_error_function(variable_values, template):

    myrank = ezff.get_pool_rank()

    # Configure LAMMPS job.
    path = str(myrank)
    md_job = lammps.job(path=path)
    md_job.structure = gt_structure
    md_job.forcefield = md_job.generate_forcefield(template, variable_values, FFtype='LJ')
    md_job.options['pbc'] = True
    md_job.options['relax_atoms'] = True
    md_job.options['relax_cell'] = True

    # Submit job and read output
    md_job.run_elastic()

    # Calculate errors in lattice constant and elastic modulus
    moduli = md_job.read_elastic_moduli()                                                # 6X6 elastic modulus tensor inside a list of length 1 (for a single input structure)
    try:
        md_bulk_modulus = np.mean([moduli[0][i,i] for i in range(3)])                        # Bulk modulus is sum of diagonal elements in moduli[0]
        bulk_modulus_error = np.linalg.norm(md_bulk_modulus - gt_bulk_modulus)               # Error is the L2 norm of deviation from the ground truth value
    except:
        bulk_modulus_error = np.nan

    md_structure = md_job.read_structure()                                               # Read relaxed structure after optimization
    error_abc, error_ang = ezff.error_lattice_constant(MD=md_structure, GT=gt_structure) # error_abc = error in lattice constants, error_ang = error in lattice angles
    latt_error_1 = np.linalg.norm(error_abc[0])                                            # error in 'a' lattice constant
    latt_error_2 = np.linalg.norm(error_abc[1])                                            # error in 'a' lattice constant

    #md_job.cleanup()

    return [latt_error_1+latt_error_2, bulk_modulus_error]


if __name__ == '__main__':

    obj = ezff.FFParam(error_function = my_error_function, num_errors = 2)
    obj.read_variable_bounds('variable_bounds')
    obj.read_forcefield_template('template')

    pool = obj.generate_pool('multi')

    obj.set_algorithm('randomsearch_so', population_size = 20)
    obj.parameterize(num_epochs = 20, pool = pool)
    obj.set_algorithm('NSGA2_MO_PLATYPUS', population_size = 20)
    obj.parameterize(num_epochs = 10, pool = pool)
