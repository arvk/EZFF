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
import nevergrad as ng
import mobopt

__version__ = '0.9.4' # Update setup.py if version changes


class FFParam(object):
    """
    Class for EZFF Forcefield Parametrization
    """

    def __init__(self, error_function=None, num_errors=None):
        """
        :param num_errors: Number of errors to be minimized for forcefield optimization
        :type num_errors: int

        :param error_function: User-defined function that takes-in a dictionary of variable-value pairs and outputs a list of computed errors
        :type error_function: function
        """
        self.error_function = error_function
        self.num_errors = num_errors
        self.relative_weights = np.array([1.0 for i in range(num_errors)])
        self.variables = []
        self.errors = []

    def read_variable_bounds(self, filename):
        """Read permissible lower and upper bounds for decision variables used in forcefields optimization

        :param filename: Name of text file listing bounds for each decision variable that must be optimized
        :type filename: str
        """
        self.variable_bounds = ffio.read_variable_bounds(filename)
        self.num_variables = len(self.variable_bounds.keys())
        self.variable_names = [key for key in self.variable_bounds.keys()]


    def read_forcefield_template(self, template_filename):
        """Read-in the forcefield template. The template is constructed from a functional forcefield file by replacing all optimizable numerical values with variable names enclosed within dual angled brackets << and >>.

        :param template_filename: Name of the forcefield template file to be read-in
        :type template_filename: str
        """
        self.forcefield_template = ffio.read_forcefield_template(template_filename)

    def set_algorithm(self, algo_string, population_size = None):
        """
        Set optimization algorithm. Initialize interfaces to external optimizers and return the algorithm object

        :param algo_string: Type of algorithm to parameterize the forcefield
        :type algo_string: str

        :param population_size: Number of candidate forcefields evaluated every epoch
        :type algo_string: int
        """

        self.population_size = population_size
        self.algo_string = algo_string

        ng_algos = ['NGOPT_SO', 'TWOPOINTSDE_SO','PORTFOLIODISCRETEONEPLUSONE_SO','ONEPLUSONE_SO','CMA_SO','TBPSA_SO', 'PSO_SO', 'SCRHAMMERSLEYSEARCHPLUSMIDDLEPOINT_SO', 'RANDOMSEARCH_SO']
        mobopt_algos = ['MOBO']

        if algo_string.upper() in ng_algos:
            self.algo_framework = 'nevergrad'
            ng_variable_dict = ng.p.Dict()
            for variable in self.variable_bounds.keys():
                ng_variable_dict[variable] = ng.p.Scalar(lower = self.variable_bounds[variable][0], upper = self.variable_bounds[variable][1])

            if algo_string.upper() == 'NGOPT_SO':
                self.algorithm = ng.optimizers.NGOpt(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)
            elif algo_string.upper() == 'TWOPOINTSDE_SO':
                self.algorithm = ng.optimizers.TwoPointsDE(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)
            elif algo_string.upper() == 'PORTFOLIODISCRETEONEPLUSONE_SO':
                self.algorithm = ng.optimizers.PortfolioDiscreteOnePlusOne_SO(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)
            elif algo_string.upper() == 'ONEPLUSONE_SO':
                self.algorithm = ng.optimizers.OnePlusOne(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)
            elif algo_string.upper() == 'CMA_SO':
                self.algorithm = ng.optimizers.CMA(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)
            elif algo_string.upper() == 'TBPSA_SO':
                self.algorithm = ng.optimizers.TBPSA(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)
            elif algo_string.upper() == 'PSO_SO':
                self.algorithm = ng.optimizers.PSO(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)
            elif algo_string.upper() == 'SCRHAMMERSLEYSEARCHPLUSMIDDLEPOINT_SO':
                self.algorithm = ng.optimizers.ScrHammersleySearchPlusMiddlePoint(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)
            elif algo_string.upper() == 'RANDOMSEARCH_SO':
                self.algorithm = ng.optimizers.RandomSearch(parametrization=ng_variable_dict, budget=self.population_size, num_workers=2)

        elif algo_string.upper() in mobopt_algos:
            self.algo_framework = 'mobopt'
            var_mins = []
            var_maxs = []
            for variable in self.variable_bounds.keys():
                var_mins.append(self.variable_bounds[variable][0])
                var_maxs.append(self.variable_bounds[variable][1])
            self.mobopt_variable_bounds = np.vstack((var_mins, var_maxs)).T
            self.algorithm = mobopt.MOBayesianOpt(target = self.error_function, NObj = self.num_errors, pbounds = self.mobopt_variable_bounds)


    def ask(self):
        new_variables = []
        if self.algo_framework == 'nevergrad':
            for i in range(self.population_size):
                newvar = self.algorithm.ask()
                new_var_as_list = np.array([newvar.value[self.variable_names[i]] for i in range(len(self.variable_names))])
                new_variables.append(new_var_as_list)
            return new_variables

        elif self.algo_framework == 'mobopt':
            q = 0.5
            prob = 0.1

            for pointID in range(len(self.variables)):
                self.algorithm.space.add_observation(np.array(self.variables[pointID]), 0.0 - np.array(self.errors[pointID]))

            for i in range(self.num_errors):
                yy = self.algorithm.space.f[:, i]
                self.algorithm.GP[i].fit(self.algorithm.space.x, yy)

            pop, logbook, front = mobopt._NSGA2.NSGAII(self.algorithm.NObj,
                                                       self.algorithm._MOBayesianOpt__ObjectiveGP,
                                                       self.algorithm.pbounds,
                                                       MU=self.population_size*2)


            Population = np.asarray(pop)
            IndexF, FatorF = self.algorithm._MOBayesianOpt__LargestOfLeast(front, self.algorithm.space.f)
            IndexPop, FatorPop = self.algorithm._MOBayesianOpt__LargestOfLeast(Population, self.algorithm.space.x)

            Fator = q * FatorF + (1-q) * FatorPop
            sorted_ids = np.argsort(Fator)


            for i in range(self.population_size):
                Index_try = int(np.argwhere(sorted_ids == np.max(sorted_ids)-i))
                self.algorithm.x_try = Population[Index_try]

                if self.algorithm.space.RS.uniform() < prob:
                    if self.algorithm.NParam > 1:
                        ii = self.algorithm.space.RS.randint(low=0, high=self.algorithm.NParam - 1)
                    else:
                        ii = 0

                    self.algorithm.x_try[ii] = self.algorithm.space.RS.uniform(low=self.algorithm.pbounds[ii][0],high=self.algorithm.pbounds[ii][1])

                new_variables.append(self.algorithm.x_try)

            return new_variables



    def parameterize(self, num_epochs = None):

        if self.algo_framework == 'nevergrad':
            for epoch in range(num_epochs):

                self.set_algorithm(algo_string = self.algo_string, population_size = self.population_size)

                for computed_id, computed in enumerate(self.variables):
                    computed_value = {k:v for (k,v) in zip(self.variable_names,computed)}
                    self.algorithm.suggest(computed_value)
                    asked_suggestion = self.algorithm.ask()
                    self.algorithm.tell(asked_suggestion, self.errors[computed_id])

                new_variables = self.ask()
                new_errors = []
                for variable in new_variables:
                    variable_dict = dict(zip(self.variable_names, variable))
                    error = self.error_function(variable_dict)
                    new_errors.append(error)

                for variable_id, variable in enumerate(new_variables):
                    self.variables.append(variable)
                    self.errors.append(new_errors[variable_id])


        elif self.algo_framework == 'mobopt':
            for epoch in range(num_epochs):

                self.set_algorithm(algo_string = self.algo_string, population_size = self.population_size)

                new_variables = self.ask()
                new_errors = []
                for variable in new_variables:
                    variable_dict = dict(zip(self.variable_names, variable))
                    error = self.error_function(variable_dict)
                    new_errors.append(error)

                for variable_id, variable in enumerate(new_variables):
                    self.variables.append(variable)
                    self.errors.append(new_errors[variable_id])


    def get_best_recommendation(self):
        best_variables = None
        best_errors = None
        if self.algo_framework == 'nevergrad':
            best_recommendation = self.algorithm.provide_recommendation()
            best_variables = best_recommendation.value
            best_errors = best_recommendation.loss
        elif self.algo_framework == 'mobopt':
            best_errors, best_variables = self.algorithm.space.ParetoSet()
        return [best_variables, best_errors]





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
