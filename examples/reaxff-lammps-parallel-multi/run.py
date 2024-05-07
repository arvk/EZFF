import ezff
from ezff.interfaces import lammps, qchem
from ezff.utils.reaxff import reax_forcefield
import time

# Define ground truths
gt_gs = qchem.read_structure('ground_truths/optCHOSx.out')
gt_gs_energy = qchem.read_energy('ground_truths/optCHOSx.out')
gt_freq_scan = qchem.read_structure('ground_truths/frequency_length_scan/CHOSx.out')
gt_freq_scan_energy = qchem.read_energy('ground_truths/frequency_length_scan/CHOSx.out')
gt_full_scan = qchem.read_structure(['ground_truths/dissociation_length_scan/CHOSx.run1.out', 'ground_truths/dissociation_length_scan/CHOSx.run2.out', 'ground_truths/dissociation_length_scan/CHOSx.run3.out'])
gt_full_scan_energy = qchem.read_energy(['ground_truths/dissociation_length_scan/CHOSx.run1.out', 'ground_truths/dissociation_length_scan/CHOSx.run2.out', 'ground_truths/dissociation_length_scan/CHOSx.run3.out'])

def my_error_function(variable_values, template):

    myrank = ezff.get_pool_rank()

    path = str(myrank)+'/relaxed'

    # Calculate Ground State
    md_gs_job = lammps.job(path = path)
    md_gs_job.structure = gt_gs
    md_gs_job.forcefield = md_gs_job.generate_forcefield(template, variable_values, FFtype = 'reaxff')
    md_gs_job.options['pbc'] = False
    md_gs_job.options['relax_atoms'] = False
    md_gs_job.options['relax_cell'] = False
    # Run GULP calculation
    md_gs_job.run()
    # Read output from completed GULP job and clean-up
    md_gs_energy = md_gs_job.read_energy()
    #md_gs_job.cleanup()

    # Calculate PES Scan for frequency error
    md_freq_scan_job = lammps.job(path = path)
    md_freq_scan_job.structure = gt_freq_scan
    md_freq_scan_job.forcefield = md_freq_scan_job.generate_forcefield(template, variable_values, FFtype = 'reaxff')
    md_freq_scan_job.options['pbc'] = False
    md_freq_scan_job.options['relax_atoms'] = False
    md_freq_scan_job.options['relax_cell'] = False
    # Run GULP calculation
    md_freq_scan_job.run()
    # Read output from completed GULP job and clean-up
    md_freq_scan_energy = md_freq_scan_job.read_energy()
    #md_freq_scan_job.cleanup()

    # Calculate PES Scan for dissociation error
    md_full_scan_job = lammps.job(path = path)
    md_full_scan_job.structure = gt_full_scan
    md_full_scan_job.forcefield = md_full_scan_job.generate_forcefield(template, variable_values, FFtype = 'reaxff')
    md_full_scan_job.options['pbc'] = False
    md_full_scan_job.options['relax_atoms'] = False
    md_full_scan_job.options['relax_cell'] = False
    # Run GULP calculation
    md_full_scan_job.run()
    # Read output from completed GULP job and clean-up
    md_full_scan_energy = md_full_scan_job.read_energy()
    #md_full_scan_job.cleanup()


    # Calculate error
    freq_error = ezff.error_energy(md_freq_scan_energy-md_gs_energy, gt_freq_scan_energy-gt_gs_energy, weights = 'minima')
    dissociation_error = ezff.error_energy(md_full_scan_energy-md_gs_energy, gt_full_scan_energy-gt_gs_energy, weights = 'dissociation')
    return [freq_error, dissociation_error]


# Generate forcefield template and variable ranges
FF = reax_forcefield('ffield')
FF.make_template_twobody('S','C',double_bond=True)
FF.make_template_threebody('S','C','S')
FF.make_template_fourbody('S','C','S','S')
FF.generate_templates()

time.sleep(1.0)

if __name__ == '__main__':

    obj = ezff.FFParam(error_function = my_error_function, num_errors = 2)
    obj.read_variable_bounds('param_ranges')
    obj.read_forcefield_template('ff.template.generated')

    pool = obj.generate_pool('multi')

    obj.set_algorithm('randomsearch_so', population_size = 9)
    obj.parameterize(num_epochs = 5, pool = pool)
    obj.set_algorithm('NSGA2_MO_PLATYPUS', population_size = 9)
    obj.parameterize(num_epochs = 5, pool = pool)
