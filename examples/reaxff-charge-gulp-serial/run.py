import ezff
from ezff.interfaces import gulp, qchem
from ezff.utils.reaxff import reax_forcefield

# Define ground truths
gt_gs = qchem.read_structure('ground_truths/optCHOSx.out')
gt_gs_atomic_charges = qchem.read_atomic_charges('ground_truths/optCHOSx.out')

def my_error_function(variable_values, template):

    myrank = ezff.get_pool_rank()
    path = str(myrank)

    # Calculate Ground State Charges
    md_gs_job = gulp.job(path = path)
    md_gs_job.structure = gt_gs
    md_gs_job.forcefield = ezff.generate_forcefield(template, variable_values, FFtype = 'reaxff')
    md_gs_job.options['pbc'] = False
    md_gs_job.options['relax_atoms'] = False
    md_gs_job.options['relax_cell'] = False
    md_gs_job.options['atomic_charges'] = True
    # Run GULP calculation
    md_gs_job.run()
    # Read output from completed GULP job and clean-up
    md_gs_atomic_charges = md_gs_job.read_atomic_charges()
    md_gs_job.cleanup()

    # Calculate error
    charg_error = ezff.error_atomic_charges(MD=md_gs_atomic_charges, GT=gt_gs_atomic_charges)
    return [charg_error]


# Generate forcefield template and variable ranges
FF = reax_forcefield('ffield')
FF.make_template_qeq('S')
FF.generate_templates()


if __name__ == '__main__':

    obj = ezff.FFParam(error_function = my_error_function, num_errors = 1)
    obj.read_variable_bounds('param_ranges')
    obj.read_forcefield_template('ff.template.generated')

    obj.set_algorithm('randomsearch_so', population_size = 32)
    obj.parameterize(num_epochs = 5)
    obj.set_algorithm('ngopt_so', population_size = 32)
    obj.parameterize(num_epochs = 5)
