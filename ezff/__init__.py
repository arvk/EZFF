"""This module provide general functions for EZFF"""
import numpy as np
import xtal
from platypus import Problem, Real, unique, nondominated, NSGAII, NSGAIII, IBEA, PoolEvaluator
from platypus.mpipool import MPIPool
import os
import sys
from .ffio import write_forcefield_file

class Problem(Problem):
    def __init__(self, num_objectives = None, objective_function = None, variables = None, variable_bounds = None, template = None):
        super(Problem, self).__init__(len(variables),num_objectives)
        for counter, value in enumerate(variables):
            if value[0] == '_':
                self.types[counter] = Integer(variable_bounds[value][0], variable_bounds[value][1])
            else:
                self.types[counter] = Real(variable_bounds[value][0], variable_bounds[value][1])

        self.objective_function = objective_function
        self.variables = variables
        self.directions = [Problem.MINIMIZE for objective in range(num_objectives)]
        self.template = template

    def evaluate(self, solution):
        current_var_dict = dict(zip(self.variables, solution.variables))
        solution.objectives[:] = self.objective_function(current_var_dict)


class Pool(MPIPool):
    def __init__(self):
        super(Pool,self).__init__()


def error_phonon_dispersion(md_disp, gt_disp, weights='uniform', verbose=False):
    """Calculate error between MD-computed dispersion and the ground-truth"""
    # Perform sanity check. Number of bands should be equal between the two structures
    if not len(md_disp) == len(gt_disp):
        raise ValueError("MD and ground truth dispersions have different number of bands")
        return

    # Create array of weights - one value per band
    num_band = len(md_disp)
    W = np.ones(num_band)
    if weights == 'uniform':
        pass
    elif weights == 'acoustic':
        maxfreq = np.amax(gt_disp)
        W = np.reciprocal((np.mean(gt_disp,axis=1)/maxfreq) + 0.1)

    # Compute the RMS error between dispersions
    rms_error = 0.0
    num_k_gt = len(gt_disp[0])
    scaling = num_k_gt/100.0
    for band_index in range(0,len(gt_disp)):
        interp_md_band = np.interp(np.arange(0,num_k_gt),np.arange(0,100)*scaling,md_disp[band_index])
        rms_error += np.linalg.norm(interp_md_band - gt_disp[band_index]) * W[band_index]

    rms_error /= (num_k_gt * num_band)
    return rms_error


def read_atomic_structure(structure_file):
    structure = xtal.AtTraj(verbose=False)

    if ('POSCAR' in structure_file) or ('CONTCAR' in structure_file):
        structure.read_snapshot_vasp(structure_file)

    return structure


def optimize(problem, algorithm, iterations = 100, write_forcefields = None):
    if not isinstance(write_forcefields, int):
        write_forcefields = iterations
    for i in range(0,iterations):
        print('Epoch: '+ str(i))
        algorithm.step()

        # Make output files/directories
        outdir = 'results/' + str(i)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        varfilename = outdir + '/variables'
        objfilename = outdir + '/objectives'
        varfile = open(varfilename,'w')
        objfile = open(objfilename,'w')
        for solution in unique(nondominated(algorithm.result)):
            varfile.write(' '.join([str(variables) for variables in solution.variables]))
            varfile.write('\n')
            objfile.write(' '.join([str(objectives) for objectives in solution.objectives]))
            objfile.write('\n')
        varfile.close()
        objfile.close()

        if i % (write_forcefields-1) == 0:
            if not os.path.isdir(outdir+'/forcefields'):
                os.makedirs(outdir+'/forcefields')
            for sol_index, solution in enumerate(unique(nondominated(algorithm.result))):
                ff_name = outdir + '/forcefields/FF_' + str(sol_index)
                parameters_dict = dict(zip(problem.variables, solution.variables))
                write_forcefield_file(ff_name,problem.template,parameters_dict,verbose=False)



def pick_algorithm(myproblem, algorithm, population = 1024, evaluator = None):
    if algorithm.lower().upper() == 'NSGAII':
        if evaluator is None:
            return NSGAII(myproblem, population_size = population)
        else:
            return NSGAII(myproblem, population_size = population, evaluator = evaluator)
    elif algorithm.lower().upper() == 'NSGAIII':
        divisions = int(np.power(population * np.math.factorial(num_objectives-1),1.0/(num_objectives-1))) + 2 - num_objectives
        divisions = np.maximum(1,divisions)
        if evaluator is None:
            return NSGAIII(myproblem, divisions_outer = divisions)
        else:
            return NSGAIII(myproblem, divisions_outer = divisions, evaluator = evaluator)
    elif algorithm.lower().upper() == 'IBEA':
        if evaluator is None:
            return IBEA(myproblem, population_size = population)
        else:
            return IBEA(myproblem, population_size = population, evaluator = evaluator)
    else:
        raise Exception('Please enter an algorithm for optimization. NSGAII , NSGAIII , IBEA are supported')


def Algorithm(myproblem, algorithm, population = 1024, pool = None):
    if pool is None:
        algorithm = pick_algorithm(myproblem, algorithm, population = population)
    else:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        else:
            evaluator = PoolEvaluator(pool)
            algorithm = pick_algorithm(myproblem, algorithm, population = population, evaluator = evaluator)
    return algorithm

