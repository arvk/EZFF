import ezff
from ezff.interfaces import gulp, qchem
from ezff.utils.reaxff import reax_forcefield
import time

# Define ground truths
gt_gs = qchem.read_structure('ground_truths/optCHOSx.out')
gt_gs_energy = qchem.read_energy('ground_truths/optCHOSx.out')
gt_freq_scan = qchem.read_structure('ground_truths/frequency_length_scan/CHOSx.out')
gt_freq_scan_energy = qchem.read_energy('ground_truths/frequency_length_scan/CHOSx.out')
gt_full_scan = qchem.read_structure(['ground_truths/dissociation_length_scan/CHOSx.run1.out', 'ground_truths/dissociation_length_scan/CHOSx.run2.out', 'ground_truths/dissociation_length_scan/CHOSx.run3.out'])
gt_full_scan_energy = qchem.read_energy(['ground_truths/dissociation_length_scan/CHOSx.run1.out', 'ground_truths/dissociation_length_scan/CHOSx.run2.out', 'ground_truths/dissociation_length_scan/CHOSx.run3.out'])

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

    # Calculate PES Scan for frequency error
    md_freq_scan_job = gulp.job(path = path)
    md_freq_scan_job.structure = gt_freq_scan
    md_freq_scan_job.forcefield = ezff.generate_forcefield(template, rr, FFtype = 'reaxff')
    md_freq_scan_job.options['pbc'] = False
    md_freq_scan_job.options['relax_atoms'] = False
    md_freq_scan_job.options['relax_cell'] = False
    # Run GULP calculation
    md_freq_scan_job.run(command='gulp')
    # Read output from completed GULP job and clean-up
    md_freq_scan_energy = md_freq_scan_job.read_energy()
    md_freq_scan_job.cleanup()

    # Calculate PES Scan for dissociation error
    md_full_scan_job = gulp.job(path = path)
    md_full_scan_job.structure = gt_full_scan
    md_full_scan_job.forcefield = ezff.generate_forcefield(template, rr, FFtype = 'reaxff')
    md_full_scan_job.options['pbc'] = False
    md_full_scan_job.options['relax_atoms'] = False
    md_full_scan_job.options['relax_cell'] = False
    # Run GULP calculation
    md_full_scan_job.run(command='gulp')
    # Read output from completed GULP job and clean-up
    md_full_scan_energy = md_full_scan_job.read_energy()
    md_full_scan_job.cleanup()


    # Calculate error
    freq_error = ezff.error_energy(md_freq_scan_energy-md_gs_energy, gt_freq_scan_energy-gt_gs_energy, weights = 'minima')
    dissociation_error = ezff.error_energy(md_full_scan_energy-md_gs_energy, gt_full_scan_energy-gt_gs_energy, weights = 'dissociation')
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
bounds = ezff.read_variable_bounds('param_ranges', verbose=False)
template = ezff.read_forcefield_template('ff.template.generated')

problem = ezff.OptProblem(num_errors = 2, variable_bounds = bounds, error_function = my_error_function, template = template)
algorithm = ezff.Algorithm(problem, 'NSGAII', population = 128, pool = pool)
ezff.optimize(problem, algorithm, iterations = 4, write_forcefields = 5)
pool.close()
