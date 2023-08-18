import ezff
from ezff.interfaces import vasp, gulp
import numpy as np

bounds = ezff.read_variable_bounds('variable_bounds', verbose=False)
template = ezff.read_forcefield_template('template')

# DEFINE GROUND TRUTHS
gt_relax_disp_GS = vasp.read_phonon_dispersion('ground_truths/relaxed/G_S.dat')
gt_relax_structure = vasp.read_atomic_structure('ground_truths/relaxed/POSCAR')
gt_c11 = 160.0 #GPa for C11 of MoSe2

gt_expanded_disp_GS = vasp.read_phonon_dispersion('ground_truths/expanded/G_S.dat')
gt_expanded_structure = vasp.read_atomic_structure('ground_truths/expanded/POSCAR')

gt_compressed_disp_GS = vasp.read_phonon_dispersion('ground_truths/compressed/G_S.dat')
gt_compressed_structure = vasp.read_atomic_structure('ground_truths/compressed/POSCAR')


def my_error_function(variable_values, template):

    myrank = ezff.get_pool_rank()

    # FOR THE RELAXED STRUCTURE
    path = str(myrank)+'/relaxed'
    relaxed_job = gulp.job(path=path)
    relaxed_job.structure = gt_relax_structure
    relaxed_job.forcefield = ezff.generate_forcefield(template, variable_values, FFtype = 'SW')
    relaxed_job.options['pbc'] = True
    relaxed_job.options['relax_atoms'] = True
    relaxed_job.options['relax_cell'] = True
    relaxed_job.options['phonon_dispersion'] = True
    relaxed_job.options['phonon_dispersion_from'] = '0 0 0'
    relaxed_job.options['phonon_dispersion_to'] = '0.5 0.5 0'
    # Submit job and read output
    relaxed_job.run()
    # Read output from completed GULP job and cleanup job files
    md_relax_disp_GS = relaxed_job.read_phonon_dispersion()
    md_relaxed_moduli = relaxed_job.read_elastic_moduli()
    md_relaxed_structure = relaxed_job.read_structure()
    relaxed_job.cleanup()  # FINISH RELAXED JOB


    # FOR THE COMPRESSED STRUCTURE
    path = str(myrank)+'/compressed'
    compressed_job = gulp.job(path=path)
    compressed_job.structure = gt_compressed_structure
    compressed_job.forcefield = ezff.generate_forcefield(template, variable_values, FFtype = 'SW')
    compressed_job.options['pbc'] = True
    compressed_job.options['relax_atoms'] = True
    compressed_job.options['relax_cell'] = False
    compressed_job.options['phonon_dispersion'] = True
    compressed_job.options['phonon_dispersion_from'] = '0 0 0'
    compressed_job.options['phonon_dispersion_to'] = '0.5 0.5 0'
    # Submit job and read output
    compressed_job.run()
    # Read output from completed GULP job and cleanup job files
    md_compressed_disp_GS = compressed_job.read_phonon_dispersion()
    compressed_job.cleanup()  # FINISH COMPRESSED JOB


    # FOR THE EXPANDED STRUCTURE
    path = str(myrank)+'/expanded'
    expanded_job = gulp.job(path=path)
    expanded_job.structure = gt_expanded_structure
    expanded_job.forcefield = ezff.generate_forcefield(template, variable_values, FFtype = 'SW')
    expanded_job.options['pbc'] = True
    expanded_job.options['relax_atoms'] = True
    expanded_job.options['relax_cell'] = False
    expanded_job.options['phonon_dispersion'] = True
    expanded_job.options['phonon_dispersion_from'] = '0 0 0'
    expanded_job.options['phonon_dispersion_to'] = '0.5 0.5 0'
    # Submit job and read output
    expanded_job.run()
    # Read output from completed GULP job and cleanup job files
    md_expanded_disp_GS = expanded_job.read_phonon_dispersion()
    expanded_job.cleanup()  # FINISH EXPANDED JOB

    # Compute 5 errors from the 3 GULP jobs
    error_abc, error_ang = ezff.error_lattice_constant(MD=md_relaxed_structure, GT=gt_relax_structure)
    a_lattice_error = np.linalg.norm(error_abc[0])   # Error in 'a' lattice constant
    b_lattice_error = np.linalg.norm(error_abc[1])   # Error in 'b' lattice constant

    md_c11 = md_relaxed_moduli[0][0,0] * md_relaxed_structure.box[2,2] * (2.0/13.97)  # Extracting c11 for a bulk-like layered structure from the monolayer GULP calculation
    modulus_error = np.linalg.norm(md_c11 - gt_c11)

    phon_error_relaxed = ezff.error_phonon_dispersion(MD=md_relax_disp_GS, GT=gt_relax_disp_GS, weights='acoustic')
    phon_error_expanded = ezff.error_phonon_dispersion(MD=md_expanded_disp_GS, GT=gt_expanded_disp_GS, weights='acoustic')
    phon_error_compressed = ezff.error_phonon_dispersion(MD=md_compressed_disp_GS, GT=gt_compressed_disp_GS, weights='acoustic')

    return [a_lattice_error, b_lattice_error, modulus_error, phon_error_relaxed, phon_error_compressed, phon_error_expanded]



if __name__ == '__main__':

    obj = ezff.FFParam(error_function = my_error_function, num_errors = 6)
    obj.read_variable_bounds('variable_bounds')
    obj.read_forcefield_template('template')

    pool = obj.generate_pool('multi')

    obj.set_algorithm('randomsearch_so', population_size = 16)
    obj.parameterize(num_epochs = 5, pool = pool)
    obj.set_algorithm('nsga2_mo_platypus', population_size = 16)
    obj.parameterize(num_epochs = 25, pool = pool)
