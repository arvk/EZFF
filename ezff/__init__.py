"""This module provide general functions for EZFF"""
import os
import sys
import xtal
import numpy as np
from platypus import Problem, unique, nondominated, NSGAII, NSGAIII, IBEA, PoolEvaluator
from platypus.types import Real, Integer
from platypus.operators import InjectedPopulation, GAOperator, SBX, PM
try:
    from platypus.mpipool import MPIPool
except ImportError:
    pass
from .ffio import *
from .errors import *

__version__ = '0.9.4' # Update setup.py if version changes



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



def optimize(problem, algorithm, iterations=100, write_forcefields=None):
    """
    The optimize function provides a uniform wrapper to solve the EZFF problem using the algorithm(s) provided.

    :param problem: EZFF Problem to be optimized
    :type problem: Problem

    :param algorithm: EZFF Algorithm(s) to use for optimization. Allowed options are ``NSGAII``, ``NSGAIII`` and ``IBEA``, or a list containing any sequence of these options. The algorithms will be used in the sequence provided
    :type algorithm: str or list (of strings)

    :param iterations: Number of epochs to perform the optimization for. If multiple algorithms are specified, one iteration value should be provided for each algorithm
    :type iterations: int or list (of ints)

    :param write_forcefields: All non-dominated forcefields are written out every ``write_forcefields`` epochs. If this is ``None``, the forcefields are written out for the first and last epoch
    :type write_forcefields: int or None

    """
    # Convert algorithm and iterations into lists
    if not isinstance(algorithm, list):
        algorithm = [algorithm]
    if not isinstance(iterations, list):
        iterations = [iterations]

    if not len(algorithm) == len(iterations):
        raise ValueError("Please provide a maximum number of epochs for each algorithm")

    total_epochs = 0
    current_solutions = None
    for stage in range(0,len(algorithm)):

        # Construct an algorithm
        algorithm_for_this_stage = _generate_algorithm(algorithm[stage]["myproblem"],
                                                      algorithm[stage]["algorithm_string"],
                                                      algorithm[stage]["population"],
                                                      algorithm[stage]["mutation_probability"],
                                                      current_solutions,
                                                      algorithm[stage]["pool"])

        if not isinstance(write_forcefields, int):
            write_forcefields = np.sum([iterations[stage_no] for stage_no in range(stage+1)])

        for i in range(0, iterations[stage]):
            total_epochs += 1
            print('Epoch: '+ str(total_epochs))
            algorithm_for_this_stage.step()

            # Make output files/directories
            outdir = 'results/' + str(total_epochs)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            varfilename = outdir + '/variables'
            objfilename = outdir + '/errors'
            varfile = open(varfilename, 'w')
            objfile = open(objfilename, 'w')
            for solution in unique(nondominated(algorithm_for_this_stage.result)):
                varfile.write(' '.join([str(variables) for variables in solution.variables]))
                varfile.write('\n')
                objfile.write(' '.join([str(objective) for objective in solution.objectives]))
                objfile.write('\n')
            varfile.close()
            objfile.close()

            if total_epochs % write_forcefields == 0:
                if not os.path.isdir(outdir+'/forcefields'):
                    os.makedirs(outdir+'/forcefields')
                for sol_index, solution in enumerate(unique(nondominated(algorithm_for_this_stage.result))):
                    ff_name = outdir + '/forcefields/FF_' + str(sol_index+1)
                    parameters_dict = dict(zip(problem.variables, solution.variables))
                    generate_forcefield(problem.template, parameters_dict, outfile=ff_name)

            current_solutions = algorithm_for_this_stage.population



def Algorithm(myproblem, algorithm_string, population=1024, mutation_probability=None, pool=None):
    """
    Provide a uniform interface to initialize an algorithm class for serial and parallel execution

    :param myproblem: EZFF Problem to be optimized
    :type myproblem: Problem

    :param algorithm_string: EZFF Algorithm to use for optimization. Allowed options are ``NSGAII``, ``NSGAIII`` and ``IBEA``
    :type algorithm_string: str

    :param population: Population size for genetic algorithms
    :type population: int

    :param mutation_probability: Probability of a decision variable to undergo mutation to a random value within defined bounds
    :type mutation_probability: float between 0.0 and 1.0

    :param pool: MPI pool for parallel execution. If this is None, serial execution is assumed
    :type pool: MPIPool or None
    """
    return {"myproblem": myproblem, "algorithm_string": algorithm_string, "population": population, "mutation_probability": mutation_probability, "pool": pool}



def _pick_algorithm(myproblem, algorithm, population, mutation_probability, current_solution, evaluator=None):
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

    if mutation_probability is None:
        variator = GAOperator(SBX(), PM())
    else:
        variator = GAOperator(SBX(), PM(probability=mutation_probability))
        print('Using provided mutation probability')

    if algorithm.lower().upper() == 'NSGAII':
        if evaluator is None:
            if current_solution is None:
                return NSGAII(myproblem, population_size=population, variator=variator)
            else:
                return NSGAII(myproblem, population_size=population, variator=variator, generator=InjectedPopulation(current_solution[0:population]))
        else:
            if current_solution is None:
                return NSGAII(myproblem, population_size=population, variator=variator, evaluator=evaluator)
            else:
                return NSGAII(myproblem, population_size=population, variator=variator, generator=InjectedPopulation(current_solution[0:population]), evaluator=evaluator)
    elif algorithm.lower().upper() == 'NSGAIII':
        num_errors = len(myproblem.directions)
        divisions = int(np.power(population * np.math.factorial(num_errors-1), 1.0/(num_errors-1))) + 2 - num_errors
        divisions = np.maximum(1, divisions)
        if evaluator is None:
            if current_solution is None:
                return NSGAIII(myproblem, divisions_outer=divisions, variator=variator)
            else:
                return NSGAIII(myproblem, divisions_outer=divisions, variator=variator, generator=InjectedPopulation(current_solution[0:population]))
        else:
            if current_solution is None:
                return NSGAIII(myproblem, divisions_outer=divisions, variator=variator, evaluator=evaluator)
            else:
                return NSGAIII(myproblem, divisions_outer=divisions, variator=variator, generator=InjectedPopulation(current_solution[0:population]), evaluator=evaluator)
    elif algorithm.lower().upper() == 'IBEA':
        if evaluator is None:
            if current_solution is None:
                return IBEA(myproblem, population_size=population, variator=variator)
            else:
                return IBEA(myproblem, population_size=population, variator=variator, generator=InjectedPopulation(current_solution[0:population]))
        else:
            if current_solution is None:
                return IBEA(myproblem, population_size=population, variator=variator, evaluator=evaluator)
            else:
                return IBEA(myproblem, population_size=population, variator=variator, generator=InjectedPopulation(current_solution[0:population]), evaluator=evaluator)
    else:
        raise Exception('Please enter an algorithm for optimization. NSGAII , NSGAIII , IBEA are supported')



def _generate_algorithm(myproblem, algorithm_string, population, mutation_probability, current_solution, pool):
    if pool is None:
        algorithm = _pick_algorithm(myproblem, algorithm_string, population=population, mutation_probability=mutation_probability, current_solution=current_solution)
    else:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        else:
            evaluator = PoolEvaluator(pool)
            algorithm = _pick_algorithm(myproblem, algorithm_string, population=population, mutation_probability=mutation_probability, current_solution=current_solution, evaluator=evaluator)
    return algorithm
