import sys
sys.path.append('../..')
import ezff
import ezff.ffio as ffio
from ezff.interfaces import vasp, gulp
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

bounds = ffio.read_parameter_bounds('params_range', verbose=False)
variables = [key for key in bounds.keys()]
template = ffio.read_parameter_template('template')


# DEFINE GROUND TRUTHS
gt_relax_disp_GM = vasp.read_phonon_dispersion('ground_truths/relaxed/GM.dat')
gt_relax_disp_MK = vasp.read_phonon_dispersion('ground_truths/relaxed/MK.dat')
gt_relax_disp_KG = vasp.read_phonon_dispersion('ground_truths/relaxed/KG.dat')
gt_relax_structure = ezff.read_atomic_structure('ground_truths/relaxed/POSCAR')
gt_elastic_modulus = 160.0 #GPa

gt_expanded_disp_GM = vasp.read_phonon_dispersion('ground_truths/expanded/GM.dat')
gt_expanded_disp_MK = vasp.read_phonon_dispersion('ground_truths/expanded/MK.dat')
gt_expanded_disp_KG = vasp.read_phonon_dispersion('ground_truths/expanded/KG.dat')
gt_expanded_structure = ezff.read_atomic_structure('ground_truths/expanded/POSCAR')

gt_compressed_disp_GM = vasp.read_phonon_dispersion('ground_truths/compressed/GM.dat')
gt_compressed_disp_MK = vasp.read_phonon_dispersion('ground_truths/compressed/MK.dat')
gt_compressed_disp_KG = vasp.read_phonon_dispersion('ground_truths/compressed/KG.dat')
gt_compressed_structure = ezff.read_atomic_structure('ground_truths/compressed/POSCAR')


def my_error_function(rr):
    # Get rank from pool
    try:
        myrank = pool.rank
    except:
        myrank = 0

    ffio.write_forcefield_file(str(myrank)+'/HH',template,rr,verbose=False)

    # FOR THE RELAXED STRUCTURE
    path = str(myrank)+'/relaxed'
    relaxed_job = gulp.job(path=path)
    relaxed_job.forcefield = str(myrank)+'/HH'
    relaxed_job.temporary_forcefield = False
    relaxed_job.structure = gt_relax_structure
    relaxed_job.pbc = True
    relaxed_job.options['relax_atoms'] = True
    relaxed_job.options['relax_cell'] = True
    relaxed_job.options['phonon_dispersion'] = True

    # Calcaulte G-M dispersion
    relaxed_job.options['phonon_dispersion_from'] = '0 0 0'
    relaxed_job.options['phonon_dispersion_to'] = '0.5 0 0'
    relaxed_job.write_script_file()
    relaxed_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_relax_disp_GM = gulp.read_phonon_dispersion(relaxed_job.path+'/out.gulp.disp')

    # Calcaulte M-K dispersion
    relaxed_job.options['phonon_dispersion_from'] = '0.5 0 0'
    relaxed_job.options['phonon_dispersion_to'] = '1/3 1/3 0'
    relaxed_job.write_script_file()
    relaxed_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_relax_disp_MK = gulp.read_phonon_dispersion(relaxed_job.path+'/out.gulp.disp')

    # Calcaulte K-G dispersion
    relaxed_job.options['phonon_dispersion_from'] = '1/3 1/3 0'
    relaxed_job.options['phonon_dispersion_to'] = '0 0 0'
    relaxed_job.write_script_file()
    relaxed_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_relax_disp_KG = gulp.read_phonon_dispersion(relaxed_job.path+'/out.gulp.disp')

    # Compute lattice constant and elastic modulus
    moduli = gulp.read_elastic_moduli(relaxed_job.path + '/' + relaxed_job.outfile)
    lattice = gulp.read_lattice_constant(relaxed_job.path + '/' + relaxed_job.outfile)

    latt_error = np.linalg.norm(lattice['err_abc'][0:2])
    modulus_error = np.linalg.norm((moduli[0,0]*lattice['abc'][2]*2.0/13.97)-160.0)
    phon_error_GM = ezff.error_phonon_dispersion(md_relax_disp_GM, gt_relax_disp_GM, weights='uniform')
    phon_error_MK = ezff.error_phonon_dispersion(md_relax_disp_MK, gt_relax_disp_MK, weights='uniform')
    phon_error_KG = ezff.error_phonon_dispersion(md_relax_disp_KG, gt_relax_disp_KG, weights='uniform')
    phon_error_relax = np.linalg.norm([phon_error_GM, phon_error_MK, phon_error_KG])

    relaxed_job.cleanup()  # FINISH RELAXED JOB





    # FOR THE COMPRESSED STRUCTURE
    path = str(myrank)+'/compressed'
    compressed_job = gulp.job(path=path)
    compressed_job.forcefield = str(myrank)+'/HH'
    compressed_job.temporary_forcefield = False
    compressed_job.structure = gt_compressed_structure
    compressed_job.pbc = True
    compressed_job.options['relax_atoms'] = True
    compressed_job.options['relax_cell'] = False
    compressed_job.options['phonon_dispersion'] = True

    # Calcaulte G-M dispersion
    compressed_job.options['phonon_dispersion_from'] = '0 0 0'
    compressed_job.options['phonon_dispersion_to'] = '0.5 0 0'
    compressed_job.write_script_file()
    compressed_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_compressed_disp_GM = gulp.read_phonon_dispersion(compressed_job.path+'/out.gulp.disp')

    # Calcaulte M-K dispersion
    compressed_job.options['phonon_dispersion_from'] = '0.5 0 0'
    compressed_job.options['phonon_dispersion_to'] = '1/3 1/3 0'
    compressed_job.write_script_file()
    compressed_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_compressed_disp_MK = gulp.read_phonon_dispersion(compressed_job.path+'/out.gulp.disp')

    # Calcaulte K-G dispersion
    compressed_job.options['phonon_dispersion_from'] = '1/3 1/3 0'
    compressed_job.options['phonon_dispersion_to'] = '0 0 0'
    compressed_job.write_script_file()
    compressed_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_compressed_disp_KG = gulp.read_phonon_dispersion(compressed_job.path+'/out.gulp.disp')

    phon_error_GM = ezff.error_phonon_dispersion(md_compressed_disp_GM, gt_compressed_disp_GM, weights='uniform')
    phon_error_MK = ezff.error_phonon_dispersion(md_compressed_disp_MK, gt_compressed_disp_MK, weights='uniform')
    phon_error_KG = ezff.error_phonon_dispersion(md_compressed_disp_KG, gt_compressed_disp_KG, weights='uniform')
    phon_error_compressed = np.linalg.norm([phon_error_GM, phon_error_MK, phon_error_KG])

    compressed_job.cleanup()  # FINISH RELAXED JOB






    # FOR THE EXPANDED STRUCTURE
    path = str(myrank)+'/expanded'
    expanded_job = gulp.job(path=path)
    expanded_job.forcefield = str(myrank)+'/HH'
    expanded_job.temporary_forcefield = False
    expanded_job.structure = gt_expanded_structure
    expanded_job.pbc = True
    expanded_job.options['relax_atoms'] = True
    expanded_job.options['relax_cell'] = False
    expanded_job.options['phonon_dispersion'] = True

    # Calcaulte G-M dispersion
    expanded_job.options['phonon_dispersion_from'] = '0 0 0'
    expanded_job.options['phonon_dispersion_to'] = '0.5 0 0'
    expanded_job.write_script_file()
    expanded_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_expanded_disp_GM = gulp.read_phonon_dispersion(expanded_job.path+'/out.gulp.disp')

    # Calcaulte M-K dispersion
    expanded_job.options['phonon_dispersion_from'] = '0.5 0 0'
    expanded_job.options['phonon_dispersion_to'] = '1/3 1/3 0'
    expanded_job.write_script_file()
    expanded_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_expanded_disp_MK = gulp.read_phonon_dispersion(expanded_job.path+'/out.gulp.disp')

    # Calcaulte K-G dispersion
    expanded_job.options['phonon_dispersion_from'] = '1/3 1/3 0'
    expanded_job.options['phonon_dispersion_to'] = '0 0 0'
    expanded_job.write_script_file()
    expanded_job.run(command='/staging/pv/kris658/SOFTWARE/gulp/Serial/CODE/Src/gulp')
    md_expanded_disp_KG = gulp.read_phonon_dispersion(expanded_job.path+'/out.gulp.disp')

    phon_error_GM = ezff.error_phonon_dispersion(md_expanded_disp_GM, gt_expanded_disp_GM, weights='uniform')
    phon_error_MK = ezff.error_phonon_dispersion(md_expanded_disp_MK, gt_expanded_disp_MK, weights='uniform')
    phon_error_KG = ezff.error_phonon_dispersion(md_expanded_disp_KG, gt_expanded_disp_KG, weights='uniform')
    phon_error_expanded = np.linalg.norm([phon_error_GM, phon_error_MK, phon_error_KG])

    expanded_job.cleanup()  # FINISH EXPANDED JOB


    return [latt_error, modulus_error, phon_error_relax, phon_error_compressed, phon_error_expanded]


pool = ezff.Pool()
problem = ezff.Problem(variables = variables, num_errors = 5, variable_bounds = bounds, error_function = my_error_function, template = template)
algorithm = ezff.Algorithm(problem, 'NSGAII', population = 128, pool = pool)
ezff.optimize(problem, algorithm, iterations = 10)
pool.close()
