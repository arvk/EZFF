import sys
sys.path.append('../..')
import ezff
import ezff.ffio as ffio
from ezff.interfaces import gulp, qchem
import xtal
import numpy as np
from ezff.utils import reax_forcefield
import time
import logging
logging.basicConfig(level=logging.INFO)

def convert_reax_forcefield(oldfile):
    newfile = oldfile+'.lib'
    reax_forcefield(oldfile).write_gulp_library(newfile)
    return newfile

def my_error_function(rr):
    # Get a unique path for GULP jobs from the MPI rank. Set to '0' for serial jobs
    try:
        path = str(pool.rank)
    except:
        path = '0'

    # Calculate_Ground_State
    gt_gs = qchem.read_ground_state('ground_truths/optCHOSx.out')
    gt_gs_energy = gt_gs.snaplist[0].energy
    md_gs_job = gulp.job(path = path)
    ffio.write_forcefield_file(md_gs_job.path+'/FF',template,rr,verbose=False)
    md_gs_job.forcefield = md_gs_job.path + '/FF'
    md_gs_job.temporary_forcefield = False
    md_gs_job.structure = gt_gs
    md_gs_job.pbc = False
    md_gs_job.options['relax_atoms'] = False
    md_gs_job.options['relax_cell'] = False
    md_gs_job.write_script_file(convert_reax_forcefield)
    md_gs_job.run(command='gulp')
    md_gs_energy = gulp.read_energy(md_gs_job.path + '/' + md_gs_job.outfile)
    md_gs_job.cleanup()

    # Calculate Small Length Scan for Bond Frequencies
    gt_freq_scan = qchem.read_scan('ground_truths/frequency_length_scan/CHOSx.out')
    md_freq_scan_energies = []
    gt_freq_scan_energies = []
    for snapID, snapshot in enumerate(gt_freq_scan.snaplist):
        gt_freq_scan_energies.append(snapshot.energy)
        md_freq_scan_job = gulp.job(path = path)
        ffio.write_forcefield_file(md_freq_scan_job.path+'/FF',template,rr,verbose=False)
        thisstructure = xtal.AtTraj()
        thisstructure.snaplist.append(snapshot)
        md_freq_scan_job.forcefield = md_freq_scan_job.path + '/FF'
        md_freq_scan_job.temporary_forcefield = False
        md_freq_scan_job.structure = thisstructure
        md_freq_scan_job.pbc = False
        md_freq_scan_job.options['relax_atoms'] = False
        md_freq_scan_job.options['relax_cell'] = False
        md_freq_scan_job.write_script_file(convert_reax_forcefield)
        md_freq_scan_job.run(command='gulp')
        md_freq_scan_energy = gulp.read_energy(md_freq_scan_job.path + '/' + md_freq_scan_job.outfile)
        md_freq_scan_energies.append(md_freq_scan_energy)
        md_freq_scan_job.cleanup()
        del md_freq_scan_job

    # Calculate error
    md_freq_scan_energies = np.array(md_freq_scan_energies)
    gt_freq_scan_energies = np.array(gt_freq_scan_energies)
    md_gs_energy = np.array(md_gs_energy)
    gt_gs_energy = np.array(gt_gs_energy)

    freq_error = ezff.error_PES_scan( md_freq_scan_energies-md_gs_energy, gt_freq_scan_energies-gt_gs_energy, weights = 'minima')



    # Calculate Full Length Scan for Bond Dissociation Energy
    gt_dis_scan = qchem.read_scan(['ground_truths/dissociation_length_scan/CHOSx.run1.out', 'ground_truths/dissociation_length_scan/CHOSx.run2.out', 'ground_truths/dissociation_length_scan/CHOSx.run3.out'])
    md_dis_scan_energies = []
    gt_dis_scan_energies = []
    for snapID, snapshot in enumerate(gt_dis_scan.snaplist):
        gt_dis_scan_energies.append(snapshot.energy)
        md_dis_scan_job = gulp.job(path = path)
        ffio.write_forcefield_file(md_dis_scan_job.path+'/FF',template,rr,verbose=False)
        thisstructure = xtal.AtTraj()
        thisstructure.snaplist.append(snapshot)
        md_dis_scan_job.forcefield = md_dis_scan_job.path + '/FF'
        md_dis_scan_job.temporary_forcefield = False
        md_dis_scan_job.structure = thisstructure
        md_dis_scan_job.pbc = False
        md_dis_scan_job.options['relax_atoms'] = False
        md_dis_scan_job.options['relax_cell'] = False
        md_dis_scan_job.write_script_file(convert_reax_forcefield)
        md_dis_scan_job.run(command='gulp')
        md_dis_scan_energy = gulp.read_energy(md_dis_scan_job.path + '/' + md_dis_scan_job.outfile)
        md_dis_scan_energies.append(md_dis_scan_energy)
        md_dis_scan_job.cleanup()
        del md_dis_scan_job

    # Calculate error
    md_dis_scan_energies = np.array(md_dis_scan_energies)
    gt_dis_scan_energies = np.array(gt_dis_scan_energies)
    md_gs_energy = np.array(md_gs_energy)
    gt_gs_energy = np.array(gt_gs_energy)

    dis_error = ezff.error_PES_scan( md_dis_scan_energies-md_gs_energy, gt_dis_scan_energies-gt_gs_energy, weights = 'dissociation')

    return [freq_error, dis_error]

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
