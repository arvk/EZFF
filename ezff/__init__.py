"""This module provide general functions for EZFF"""
import os
import sys
import xtal
import numpy as np
from platypus import Problem, unique, nondominated, NSGAII, NSGAIII, IBEA, PoolEvaluator
from platypus.types import Real, Integer
try:
    from platypus.mpipool import MPIPool
except ImportError:
    pass
from .ffio import write_forcefield_file

__version__ = '0.9.2' # Update setup.py if version changes

class OptProblem(Problem):
    """
    Class for Forcefield optimization problem. Derived from the generic Platypus Problem class for optimization problems
    """
    def __init__(self, num_errors=None, error_function=None, variable_bounds=None, template=None):
        """
        :param num_errors: Number of errors to be minimized for forcefield optimization
        :type num_errors: int

        :param error_function: User-defined function that takes-in a dictionary of variable-value pairs and outputs a list of computed errors
        :type error_function: function

        :param variable_bounds: Dictionary of bounds for decision variables in the format `variable: [lower_bound upper_bound]`
        :type variable_bounds: dict

        :param template: Forcefield template
        :type template: str
        """
        variables = [key for key in variable_bounds.keys()]
        super(OptProblem, self).__init__(len(variables), num_errors)
        for counter, value in enumerate(variables):
            if value[0] == '_':
                self.types[counter] = Integer(variable_bounds[value][0], variable_bounds[value][1])
            else:
                self.types[counter] = Real(variable_bounds[value][0], variable_bounds[value][1])

        self.error_function = error_function
        self.directions = [Problem.MINIMIZE for error in range(num_errors)]
        self.variables = variables
        self.template = template

    def evaluate(self, solution):
        """
        Wrapper for platypus.Problem.evaluate . Takes the array of current decision variables and repackages it as a dictionary for the error function

        :param solution: 1-D array of variables for current epoch and rank
        :type solution: list
        """
        current_var_dict = dict(zip(self.variables, solution.variables))
        solution.objectives[:] = self.error_function(current_var_dict)


class Pool(MPIPool):
    """
    Wrapper for platypus.MPIPool

    :param MPIPool: MPI Pool
    :type MPIPool: MPIPool
    """
    def __init__(self, comm=None, debug=False, loadbalance=False):
        super(Pool, self).__init__(comm=comm, debug=debug, loadbalance=loadbalance)


def error_phonon_dispersion(md_disp, gt_disp, weights='uniform', verbose=False):
    """
    Calculate error between MD-computed phonon dispersion and the ground-truth phonon dispersion with user-defined weighting schemes

    :param md_disp: MD-computed phonon dispersion curve
    :type md_disp: 2D np.array

    :param gt_disp: Ground-truth phonon dispersion curve
    :type gt_disp: 2D np.array

    :param weights: User-defined weighting scheme for calculating errors provided as a list of numbers, one per band. Possible values are
                    ``uniform`` - where errors from all bands are equally weighted,
                    ``acoustic`` - where errors from lower-frequency bands are assigned greater weights, and
                    `list` - 1-D list of length equal to number of bands
    :type weights: str `or` list

    :param verbose: Deprecated option for verbosity of error calculation routine
    :type verbose: bool
    """
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
        W = np.reciprocal((np.mean(gt_disp, axis=1)/maxfreq) + 0.1)

    # Compute the RMS error between dispersions
    rms_error = 0.0
    num_k_gt = len(gt_disp[0])
    scaling = num_k_gt/100.0
    for band_index in range(0, len(gt_disp)):
        interp_md_band = np.interp(np.arange(0, num_k_gt), np.arange(0, 100)*scaling, md_disp[band_index])
        rms_error += np.linalg.norm(interp_md_band - gt_disp[band_index]) * W[band_index]

    rms_error /= (num_k_gt * num_band)
    return rms_error



def error_PES_scan(md_scan, gt_scan, weights='uniform', verbose=False):
    """
    Calculate error between MD-computed potential energy surface and the ground-truth potential energy surface with user-defined weighting schemes

    :param md_disp: MD-computed potential energy surface
    :type md_disp: 1D np.array

    :param gt_disp: Ground-truth potential energy surface
    :type gt_disp: 1D np.array

    :param weights: User-defined weighting scheme for calculating errors. Possible values are
                    ``uniform`` - where errors from all points on the PES are weighted equally,
                    ``minima`` - where errors from lower-energy points are assigned greater weights,
                    ``dissociation`` - where errors from highest-energy points are assigned greater weights, and
                    `list` - 1-D list of length equal to number of points on the PES scans
    :type weights: str `or` list

    :param verbose: Deprecated option for verbosity of error calculation routine
    :type verbose: bool
    """
    # Perform sanity check. Number of bands should be equal between the two structures
    if not len(md_scan) == len(gt_scan):
        raise ValueError("MD and ground truth PES have different number of points! Aborting")
        return

    md_scan = np.array(md_scan)
    gt_scan = np.array(gt_scan)

    num_pes = len(gt_scan)
    W = np.ones(num_pes)
    if weights == 'uniform':
        pass
    elif weights == 'minima':
        min_E = np.amin(gt_scan)
        max_E = np.amax(gt_scan)
        W = np.reciprocal(((gt_scan-min_E)/max_E) + 0.1)
    elif weights == 'dissociation':
        min_E = np.amin(gt_scan)
        max_E = np.amax(gt_scan)
        W = (9.0*(gt_scan-min_E)/max_E) + 1.0
    elif isinstance(weights,list) or isinstance(weights,np.ndarray):
        if len(weights) == len(gt_scan):
            W = np.array(weights)
        else:
            raise ValueError("Weights array and PES have different number of points! Aborting")

    # Compute the RMS error between PES
    rms_error = np.linalg.norm((md_scan - gt_scan) * W)
    return rms_error




def read_atomic_structure(structure_file):
    """
    Read-in atomic structure. Currently only VASP POSCAR/CONTCAR files are supported

    :param structure_file: Filename of the atomic structure file
    :type structure_file: str
    """
    structure = xtal.AtTraj(verbose=False)

    if ('POSCAR' in structure_file) or ('CONTCAR' in structure_file):
        structure.read_snapshot_vasp(structure_file)

    return structure


def optimize(problem, algorithm, iterations=100, write_forcefields=None):
    """
    Uniform wrapper function that steps through the optimization process. Also provides uniform handling of output files.

    :param problem: EZFF Problem to be optimized
    :type problem: Problem

    :param algorithm: EZFF Algorithm to use for optimization. Allowed options are ``NSGAII``, ``NSGAIII`` and ``IBEA``
    :type algorithm: str

    :param iterations: Number of epochs to perform the optimization for
    :type iterations: int

    :param write_forcefields: All non-dominated forcefields are written out every ``write_forcefields`` epochs. If this is ``None``, the forcefields are written out for the first and last epoch
    :type write_forcefields: int or None

    """
    if not isinstance(write_forcefields, int):
        write_forcefields = iterations
    for i in range(0, iterations):
        print('Epoch: '+ str(i))
        algorithm.step()

        # Make output files/directories
        outdir = 'results/' + str(i)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        varfilename = outdir + '/variables'
        objfilename = outdir + '/errors'
        varfile = open(varfilename, 'w')
        objfile = open(objfilename, 'w')
        for solution in unique(nondominated(algorithm.result)):
            varfile.write(' '.join([str(variables) for variables in solution.variables]))
            varfile.write('\n')
            objfile.write(' '.join([str(objective) for objective in solution.objectives]))
            objfile.write('\n')
        varfile.close()
        objfile.close()

        if i % (write_forcefields-1) == 0:
            if not os.path.isdir(outdir+'/forcefields'):
                os.makedirs(outdir+'/forcefields')
            for sol_index, solution in enumerate(unique(nondominated(algorithm.result))):
                ff_name = outdir + '/forcefields/FF_' + str(sol_index)
                parameters_dict = dict(zip(problem.variables, solution.variables))
                write_forcefield_file(ff_name, problem.template, parameters_dict, verbose=False)



def pick_algorithm(myproblem, algorithm, population=1024, evaluator=None):
    """
    Return a serial or parallel platypus.Algorithm object based on input string

    :param myproblem: EZFF Problem to be optimized
    :type myproblem: Problem

    :param algorithm: EZFF Algorithm to use for optimization. Allowed options are ``NSGAII``, ``NSGAIII`` and ``IBEA``
    :type algorithm: str

    :param population: Population size for genetic algorithms
    :type population: int

    :param evaluator: Platypus.Evaluator in case of parallel execution
    :type evaluator: Platypus.Evaluator
    """

    if algorithm.lower().upper() == 'NSGAII':
        if evaluator is None:
            return NSGAII(myproblem, population_size=population)
        else:
            return NSGAII(myproblem, population_size=population, evaluator=evaluator)
    elif algorithm.lower().upper() == 'NSGAIII':
        divisions = int(np.power(population * np.math.factorial(num_errors-1), 1.0/(num_errors-1))) + 2 - num_errors
        divisions = np.maximum(1, divisions)
        if evaluator is None:
            return NSGAIII(myproblem, divisions_outer=divisions)
        else:
            return NSGAIII(myproblem, divisions_outer=divisions, evaluator=evaluator)
    elif algorithm.lower().upper() == 'IBEA':
        if evaluator is None:
            return IBEA(myproblem, population_size=population)
        else:
            return IBEA(myproblem, population_size=population, evaluator=evaluator)
    else:
        raise Exception('Please enter an algorithm for optimization. NSGAII , NSGAIII , IBEA are supported')


def Algorithm(myproblem, algorithm_string, population=1024, pool=None):
    """
    Provide a uniform interface to initialize an algorithm class for serial and parallel execution

    :param myproblem: EZFF Problem to be optimized
    :type myproblem: Problem

    :param algorithm_string: EZFF Algorithm to use for optimization. Allowed options are ``NSGAII``, ``NSGAIII`` and ``IBEA``
    :type algorithm_string: str

    :param population: Population size for genetic algorithms
    :type population: int

    :param pool: MPI pool for parallel execution. If this is None, serial execution is assumed
    :type pool: MPIPool or None
    """
    if pool is None:
        algorithm = pick_algorithm(myproblem, algorithm_string, population=population)
    else:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        else:
            evaluator = PoolEvaluator(pool)
            algorithm = pick_algorithm(myproblem, algorithm_string, population=population, evaluator=evaluator)
    return algorithm
