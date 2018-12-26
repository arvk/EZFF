import ffio
import interface_gulp as gulp
import interface_qchem as qchem
import ezff
import numpy as np
from platypus import NSGAII
import xtal
import os

r = ffio.read_parameter_bounds('params_range', verbose=False)
variables = [key for key in r.keys()]
template = ffio.read_parameter_template('template')

def convert_reax_forcefield(oldfile):
    newfile = oldfile+'.lib'
    os.system('./ffield2gulp < ' + oldfile + ' > ' + newfile)
    return newfile

def my_obj_function(rr):
    ffio.write_forcefield_file('HH',template,rr,verbose=False)

    # Calculate_Ground_State
    gt_gs = qchem.read_ground_state('optCHOSx.out')
    gt_gs_energy = gt_gs.snaplist[0].energy
    md_gs_job = gulp.job()
    md_gs_job.forcefield = 'HH'
    md_gs_job.temporary_forcefield = False
    md_gs_job.structure = gt_gs
    md_gs_job.pbc = False
    md_gs_job.options['relax_atoms'] = False
    md_gs_job.options['relax_cell'] = False
    md_gs_job.write_script_file(convert_reax_forcefield)
    md_gs_job.run(timeout = 6)
    md_gs_energy = gulp.read_energy(md_gs_job.outfile)
    md_gs_job.cleanup()

    # Calculate Length Scan
    gt_scan = qchem.read_scan('scanCHOSx.out')
    md_scan_energies = []
    gt_scan_energies = []
    for snapID, snapshot in enumerate(gt_scan.snaplist):
        gt_scan_energies.append(snapshot.energy)
        md_scan_job = gulp.job()
        thisstructure = xtal.AtTraj()
        thisstructure.snaplist.append(snapshot)
        md_scan_job.forcefield = 'HH'
        md_scan_job.temporary_forcefield = False
        md_scan_job.structure = thisstructure
        md_scan_job.pbc = False
        md_scan_job.options['relax_atoms'] = False
        md_scan_job.options['relax_cell'] = False
        md_scan_job.write_script_file(convert_reax_forcefield)
        md_scan_job.run(timeout = 6)
        md_scan_energy = gulp.read_energy(md_scan_job.outfile)
        md_scan_energies.append(md_scan_energy)
        md_scan_job.cleanup()
        del md_scan_job

    # Calculate error
    md_scan_energies = np.array(md_scan_energies)
    gt_scan_energies = np.array(gt_scan_energies)
    md_gs_energy = np.array(md_gs_energy)
    gt_gs_energy = np.array(gt_gs_energy)
    total_error = np.linalg.norm( (md_scan_energies-md_gs_energy) - (gt_scan_energies-gt_gs_energy)     )

    return [total_error]


myproblem = ezff.F3(variables = variables, num_objectives = 1, variable_bounds = r, objective_function = my_obj_function)
algorithm = NSGAII(myproblem, population_size=5)
algorithm.run(5)
for solution in algorithm.result:
    print(solution.variables)
