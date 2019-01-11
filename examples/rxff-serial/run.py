import sys
sys.path.append('../..')
import ezff
import ezff.ffio as ffio
from ezff.interfaces import gulp, qchem
import xtal
import numpy as np
from ezff.utils import reax_forcefield
import logging
logging.basicConfig(level=logging.INFO)

# Generate forcefield template and variable ranges
FF = reax_forcefield('ffield')
FF.make_template_twobody('S','C',double_bond=True)
FF.make_template_threebody('S','C','S')
FF.make_template_fourbody('S','C','S','S')
FF.generate_templates()

# Read template and variable ranges
bounds = ffio.read_parameter_bounds('param_ranges', verbose=False)
variables = [key for key in bounds.keys()]
template = ffio.read_parameter_template('ff.template.generated')

def convert_reax_forcefield(oldfile):
    newfile = oldfile+'.lib'
    reax_forcefield(oldfile).write_gulp_library(newfile)
    return newfile

def my_error_function(rr):
    ffio.write_forcefield_file('./0/HH',template,rr,verbose=False)

    # Calculate_Ground_State
    gt_gs = qchem.read_ground_state('ground_truths/optCHOSx.out')
    gt_gs_energy = gt_gs.snaplist[0].energy
    md_gs_job = gulp.job(path = './0/')
    md_gs_job.forcefield = md_gs_job.path + 'HH'
    md_gs_job.temporary_forcefield = False
    md_gs_job.structure = gt_gs
    md_gs_job.pbc = False
    md_gs_job.options['relax_atoms'] = False
    md_gs_job.options['relax_cell'] = False
    md_gs_job.write_script_file(convert_reax_forcefield)
    md_gs_job.run(timeout = 6)
    md_gs_energy = gulp.read_energy(md_gs_job.path + md_gs_job.outfile)
    md_gs_job.cleanup()

    # Calculate Length Scan
    gt_scan = qchem.read_scan('ground_truths/scanCHOSx.out')
    md_scan_energies = []
    gt_scan_energies = []
    for snapID, snapshot in enumerate(gt_scan.snaplist):
        gt_scan_energies.append(snapshot.energy)
        md_scan_job = gulp.job(path = './0/')
        thisstructure = xtal.AtTraj()
        thisstructure.snaplist.append(snapshot)
        md_scan_job.forcefield = md_scan_job.path + 'HH'
        md_scan_job.temporary_forcefield = False
        md_scan_job.structure = thisstructure
        md_scan_job.pbc = False
        md_scan_job.options['relax_atoms'] = False
        md_scan_job.options['relax_cell'] = False
        md_scan_job.write_script_file(convert_reax_forcefield)
        md_scan_job.run(timeout = 6)
        md_scan_energy = gulp.read_energy(md_scan_job.path + md_scan_job.outfile)
        md_scan_energies.append(md_scan_energy)
        md_scan_job.cleanup()
        del md_scan_job

    # Calculate error
    md_scan_energies = np.array(md_scan_energies)
    gt_scan_energies = np.array(gt_scan_energies)
    md_gs_energy = np.array(md_gs_energy)
    gt_gs_energy = np.array(gt_gs_energy)
    total_error = np.linalg.norm( (md_scan_energies-md_gs_energy) - (gt_scan_energies-gt_gs_energy) )

    return [total_error]


problem = ezff.Problem(variables = variables, num_errors = 1, variable_bounds = bounds, error_function = my_error_function, template = template)
algorithm = ezff.Algorithm(problem, 'NSGAII', population = 4)
ezff.optimize(problem, algorithm, iterations = 10)
