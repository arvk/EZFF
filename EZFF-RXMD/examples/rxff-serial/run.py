import ezff
from ezff.interfaces import gulp, qchem

# Define ground truths
gt_gs = qchem.read_structure('ground_truths/optCHOSx.out')
gt_gs_energy = qchem.read_energy('ground_truths/optCHOSx.out')
gt_scan = qchem.read_structure('ground_truths/scanCHOSx.out')
gt_scan_energy = qchem.read_energy('ground_truths/scanCHOSx.out')

def my_error_function(rr):
    # Get a unique path for GULP jobs from the MPI rank. Set to '0' for serial jobs
    try:
        path = str(pool.rank)
    except:
        path = '0'

    # Calculate Ground State
    md_gs_job = gulp.job(path = path)
    md_gs_job.structure = gt_gs
    md_gs_job.forcefield = ezff.generate_forcefield(template, rr, FFtype = 'reaxff')
    md_gs_job.options['pbc'] = False
    md_gs_job.options['relax_atoms'] = False
    md_gs_job.options['relax_cell'] = False
    # Run GULP calculation
    md_gs_job.run(command='gulp')
    # Read output from completed GULP job and clean-up
    md_gs_energy = md_gs_job.read_energy()
    md_gs_job.cleanup()


    # Calculate PES Scan
    md_scan_job = gulp.job(path = path)
    md_scan_job.structure = gt_scan
    md_scan_job.forcefield = ezff.generate_forcefield(template, rr, FFtype = 'reaxff')
    md_scan_job.options['pbc'] = False
    md_scan_job.options['relax_atoms'] = False
    md_scan_job.options['relax_cell'] = False
    # Run GULP calculation
    md_scan_job.run(command='gulp')
    # Read output from completed GULP job and clean-up
    md_scan_energy = md_scan_job.read_energy()
    md_scan_job.cleanup()

    # Calculate error
    total_error = ezff.error_energy( md_scan_energy-md_gs_energy, gt_scan_energy-gt_gs_energy, weights = 'uniform')
    return [total_error]

# Read template and variable ranges
bounds = ezff.read_variable_bounds('variable_bounds', verbose=False)
template = ezff.read_forcefield_template('template')

problem = ezff.OptProblem(num_errors = 1, variable_bounds = bounds, error_function = my_error_function, template = template)
algorithm = ezff.Algorithm(problem, 'NSGAII', population = 16)
ezff.optimize(problem, algorithm, iterations = 5)
