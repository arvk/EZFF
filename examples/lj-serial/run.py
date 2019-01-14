import sys
sys.path.append('../..')
import ezff
import ezff.ffio as ffio
from ezff.interfaces import gulp
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

bounds = ffio.read_variable_bounds('variable_bounds', verbose=False)
template = ffio.read_forcefield_template('template')

# DEFINE GROUND TRUTHS
gt_bulk_modulus = 1.1236 #GPa
gt_structure = ezff.read_atomic_structure('ground_truths/POSCAR')

def my_error_function(variable_values):
    # Get rank from pool
    try:
        myrank = pool.rank
    except:
        myrank = 0

    # FOR THE RELAXED STRUCTURE
    path = str(myrank)
    relaxed_job = gulp.job(path=path)
    ffio.write_forcefield_file(str(myrank)+'/FF', template, variable_values, verbose=False)
    relaxed_job.forcefield = str(myrank)+'/FF'
    relaxed_job.temporary_forcefield = False
    relaxed_job.structure = gt_structure
    relaxed_job.pbc = True
    relaxed_job.options['relax_atoms'] = True
    relaxed_job.options['relax_cell'] = True

    # Submit job and read output
    relaxed_job.write_script_file()
    relaxed_job.run(command='gulp')

    # Calculate errors in lattice constant and elastic modulus
    moduli = gulp.read_elastic_moduli(relaxed_job.path + '/' + relaxed_job.outfile)
    md_bulk_modulus = np.mean([moduli[i,i] for i in range(3)])
    lattice = gulp.read_lattice_constant(relaxed_job.path + '/' + relaxed_job.outfile)
    bulk_modulus_error = np.linalg.norm(md_bulk_modulus - gt_bulk_modulus)
    latt_error = np.linalg.norm(lattice['err_abc'][0])
    #relaxed_job.cleanup()  # FINISH RELAXED JOB

    return [latt_error, bulk_modulus_error]


problem = ezff.OptProblem(num_errors = 2, variable_bounds = bounds, error_function = my_error_function, template = template)
algorithm = ezff.Algorithm(problem, 'NSGAII', population = 32)
ezff.optimize(problem, algorithm, iterations = 16)
