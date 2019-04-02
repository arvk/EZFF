import ezff
from ezff.interfaces import gulp, qchem
import time

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
    freq_error = ezff.error_energy(md_scan_energy-md_gs_energy, gt_scan_energy-gt_gs_energy, weights = 'minima')
    dissociation_error = ezff.error_energy(md_scan_energy-md_gs_energy, gt_scan_energy-gt_gs_energy, weights = 'dissociation')
    return [freq_error, dissociation_error]




pool = ezff.Pool()

if pool.is_master():
    # Generate forcefield template and variable ranges
    FF = reax_forcefield('ffield')
    FF.make_template_twobody('S','C',double_bond=True)
    FF.make_template_threebody('S','C','S')
    FF.make_template_fourbody('S','C','S','S')
    FF.generate_templates()

time.sleep(5.0)

# Read template and variable ranges
bounds = ffio.read_variable_bounds('param_ranges', verbose=False)
template = ffio.read_forcefield_template('ff.template.generated')

problem = ezff.OptProblem(num_errors = 2, variable_bounds = bounds, error_function = my_error_function, template = template)
algorithm = ezff.Algorithm(problem, 'NSGAII', population = 128, pool = pool)
ezff.optimize(problem, algorithm, iterations = 4, write_forcefields = 5)
pool.close()
