import ffio
import interface_gulp as gulp
import interface_vasp as vasp
import ezff
import numpy as np
from platypus import NSGAII

r = ffio.read_parameter_bounds('params_range', verbose=False)
variables = [key for key in r.keys()]
template = ffio.read_parameter_template('template')

def my_obj_function(rr):
    ffio.write_forcefield_file('HH',template,rr,verbose=False)
    myjob = gulp.job()
    myjob.forcefield = 'HH'
    myjob.temporary_forcefield = True
    myjob.structure = myjob.read_atomic_structure('POSCAR')
    myjob.pbc = True
    myjob.options['relax_atoms'] = True
    myjob.options['relax_cell'] = True
    myjob.options['phonon_dispersion'] = True
    myjob.options['phonon_dispersion_from'] = '0 0 0'
    myjob.options['phonon_dispersion_to'] = '0.5 0 0'
    myjob.write_script_file()
    myjob.run(timeout = 6)

    moduli = gulp.read_elastic_moduli(myjob.outfile)
    lattice = gulp.read_lattice_constant(myjob.outfile)
    md_dispersion = gulp.read_phonon_dispersion('out.gulp.disp')
    gt_dispersion = vasp.read_phonon_dispersion('gt_band.dat')

    latt_error = np.linalg.norm(lattice['err_abc'][0:2])
    modulus_error = np.linalg.norm((moduli[0,0]*lattice['abc'][2]*2.0/13.97)-160.0)
    phon_error = ezff.error_phonon_dispersion(md_dispersion, gt_dispersion, weights='uniform')
    myjob.cleanup()

    return [latt_error, modulus_error, phon_error]


myproblem = ezff.F3(variables = variables, num_objectives = 3, variable_bounds = r, objective_function = my_obj_function)
algorithm = NSGAII(myproblem, population_size=50)
algorithm.run(5)
for solution in algorithm.result:
    print(solution.objectives)
