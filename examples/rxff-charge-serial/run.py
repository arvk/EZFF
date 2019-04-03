import ezff
from ezff.interfaces import gulp, qchem
from ezff.utils.reaxff import reax_forcefield

# Define ground truths
gt_gs = qchem.read_structure('ground_truths/optCHOSx.out')
gt_gs_atomic_charges = qchem.read_atomic_charges('ground_truths/optCHOSx.out')

def my_error_function(rr):
    # Get a unique path for GULP jobs from the MPI rank. Set to '0' for serial jobs
    try:
        path = str(pool.rank)
    except:
        path = '0'

    # Calculate Ground State Charges
    md_gs_job = gulp.job(path = path)
    md_gs_job.structure = gt_gs
    md_gs_job.forcefield = ezff.generate_forcefield(template, rr, FFtype = 'reaxff')
    md_gs_job.options['pbc'] = False
    md_gs_job.options['relax_atoms'] = False
    md_gs_job.options['relax_cell'] = False
    md_gs_job.options['atomic_charges'] = True
    # Run GULP calculation
    md_gs_job.run(command='gulp')
    # Read output from completed GULP job and clean-up
    md_gs_atomic_charges = md_gs_job.read_atomic_charges()
    md_gs_job.cleanup()

    # Calculate Relaxation
    md_relax_job = gulp.job(path = path)
    md_relax_job.structure = gt_gs
    md_relax_job.forcefield = ezff.generate_forcefield(template, rr, FFtype = 'reaxff')
    md_relax_job.options['pbc'] = False
    md_relax_job.options['relax_atoms'] = True
    md_relax_job.options['relax_cell'] = False
    md_relax_job.options['atomic_charges'] = True
    # Run GULP calculation
    md_relax_job.run(command='gulp')
    # Read output from completed GULP job and clean-up
    md_relax = md_relax_job.read_structure()
    md_relax_job.cleanup()

    # Calculate error
    charg_error = ezff.error_atomic_charges(MD=md_gs_atomic_charges, GT=gt_gs_atomic_charges)
    disp_error = ezff.error_structure_distortion(MD=md_relax, GT=gt_gs)
    return [charg_error, disp_error]


# Generate forcefield template and variable ranges
FF = reax_forcefield('ffield')
FF.make_template_qeq('S')
FF.generate_templates()

# Read template and variable ranges
bounds = ezff.read_variable_bounds('param_ranges', verbose=False)
template = ezff.read_forcefield_template('ff.template.generated')

problem = ezff.OptProblem(num_errors = 2, variable_bounds = bounds, error_function = my_error_function, template = template)
algorithm = ezff.Algorithm(problem, 'NSGAII', population = 16, mutation_probability = 0.4)
ezff.optimize(problem, algorithm, iterations = 4)
