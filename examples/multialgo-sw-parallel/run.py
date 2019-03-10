import sys
sys.path.append('../..')
import ezff
import ezff.ffio as ffio
from ezff.interfaces import vasp, gulp
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

bounds = ffio.read_variable_bounds('variable_bounds', verbose=False)
template = ffio.read_forcefield_template('template')

# DEFINE GROUND TRUTHS
gt_relaxed_disp_GS = vasp.read_phonon_dispersion('ground_truths/relaxed/G_S.dat')
gt_relaxed_structure = ezff.read_atomic_structure('ground_truths/relaxed/POSCAR')
gt_elastic_modulus = 160.0 #GPa

gt_expanded_disp_GS = vasp.read_phonon_dispersion('ground_truths/expanded/G_S.dat')
gt_expanded_structure = ezff.read_atomic_structure('ground_truths/expanded/POSCAR')

gt_compressed_disp_GS = vasp.read_phonon_dispersion('ground_truths/compressed/G_S.dat')
gt_compressed_structure = ezff.read_atomic_structure('ground_truths/compressed/POSCAR')


def my_error_function(variable_values):
    # Get rank from pool
    try:
        myrank = pool.rank
    except:
        myrank = 0

    # FOR THE RELAXED STRUCTURE
    path = str(myrank)+'/relaxed'
    relaxed_job = gulp.job(path=path)
    ffio.write_forcefield_file(str(myrank)+'/FF', template, variable_values, verbose=False)
    relaxed_job.forcefield = str(myrank)+'/FF'
    relaxed_job.temporary_forcefield = False
    relaxed_job.structure = gt_relaxed_structure
    relaxed_job.pbc = True
    relaxed_job.options['relax_atoms'] = True
    relaxed_job.options['relax_cell'] = True
    relaxed_job.options['phonon_dispersion'] = True

    # Submit job and read output
    relaxed_job.options['phonon_dispersion_from'] = '0 0 0'
    relaxed_job.options['phonon_dispersion_to'] = '0.5 0.5 0'
    relaxed_job.write_script_file()
    relaxed_job.run(command='gulp')
    md_relaxed_disp_GS = gulp.read_phonon_dispersion(relaxed_job.path+'/out.gulp.disp')
    phon_error_relax = ezff.error_phonon_dispersion(md_relaxed_disp_GS, gt_relaxed_disp_GS, weights='acoustic')

    # Calculate errors in lattice constant and elastic modulus
    moduli = gulp.read_elastic_moduli(relaxed_job.path + '/' + relaxed_job.outfile)
    lattice = gulp.read_lattice_constant(relaxed_job.path + '/' + relaxed_job.outfile)
    a_latt_error = np.linalg.norm(lattice['err_abc'][0])
    b_latt_error = np.linalg.norm(lattice['err_abc'][1])
    modulus_error = np.linalg.norm( (moduli[0,0]*lattice['abc'][2]*2.0/13.97) - gt_elastic_modulus)
    relaxed_job.cleanup()  # FINISH RELAXED JOB



    # FOR THE COMPRESSED STRUCTURE
    path = str(myrank)+'/compressed'
    compressed_job = gulp.job(path=path)
    compressed_job.forcefield = str(myrank)+'/FF'
    compressed_job.temporary_forcefield = False
    compressed_job.structure = gt_compressed_structure
    compressed_job.pbc = True
    compressed_job.options['relax_atoms'] = True
    compressed_job.options['relax_cell'] = False
    compressed_job.options['phonon_dispersion'] = True

    # Submit job and read output
    compressed_job.options['phonon_dispersion_from'] = '0 0 0'
    compressed_job.options['phonon_dispersion_to'] = '0.5 0.5 0'
    compressed_job.write_script_file()
    compressed_job.run(command='gulp')
    md_compressed_disp_GS = gulp.read_phonon_dispersion(compressed_job.path+'/out.gulp.disp')
    phon_error_compressed = ezff.error_phonon_dispersion(md_compressed_disp_GS, gt_compressed_disp_GS, weights='acoustic')
    compressed_job.cleanup()  # FINISH COMPRESSED JOB



    # FOR THE EXPANDED STRUCTURE
    path = str(myrank)+'/expanded'
    expanded_job = gulp.job(path=path)
    expanded_job.forcefield = str(myrank)+'/FF'
    expanded_job.temporary_forcefield = False
    expanded_job.structure = gt_expanded_structure
    expanded_job.pbc = True
    expanded_job.options['relax_atoms'] = True
    expanded_job.options['relax_cell'] = False
    expanded_job.options['phonon_dispersion'] = True

    # Submit job and read output
    expanded_job.options['phonon_dispersion_from'] = '0 0 0'
    expanded_job.options['phonon_dispersion_to'] = '0.5 0.5 0'
    expanded_job.write_script_file()
    expanded_job.run(command='gulp')
    md_expanded_disp_GS = gulp.read_phonon_dispersion(expanded_job.path+'/out.gulp.disp')
    phon_error_expanded = ezff.error_phonon_dispersion(md_expanded_disp_GS, gt_expanded_disp_GS, weights='acoustic')
    expanded_job.cleanup()  # FINISH EXPANDED JOB


    return [a_latt_error, b_latt_error, modulus_error, phon_error_relax, phon_error_compressed, phon_error_expanded]


pool = ezff.Pool()
problem = ezff.OptProblem(num_errors = 6, variable_bounds = bounds, error_function = my_error_function, template = template)
algo1 = ezff.Algorithm(problem, 'NSGAIII', population = 64, pool = pool)
iter1 = 10
algo2 = ezff.Algorithm(problem, 'NSGAII', population = 128, pool = pool)
iter2 = 5
algo3 = ezff.Algorithm(problem, 'IBEA', population = 32, pool = pool)
iter3 = 10
algorithms = [algo1, algo2, algo3]
iterations = [iter1, iter2, iter3]
ezff.optimize(problem, algorithms, iterations)
pool.close()
