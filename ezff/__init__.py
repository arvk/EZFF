"""This module provide general functions for EZFF"""
import os
import sys
import xtal
import numpy as np
import platypus
import mobopt
#from platypus import Problem, unique, nondominated, NSGAII, NSGAIII, IBEA, PoolEvaluator, Solution
#from platypus.types import Real, Integer
#from platypus.operators import InjectedPopulation, GAOperator, SBX, PM, TournamentSelector, RandomGenerator
#from platypus.config import default_variator
from .algorithms import *


try:
    from platypus.mpipool import MPIPool
except ImportError:
    pass
from .ffio import *
from .errors import *

__version__ = '0.9.4' # Update setup.py if version changes


class Challenge():
    def __init__(self, num_errors=None, error_function=None, variable_bounds=None, template=None, solver=None, population_size=None):
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
        self.num_errors = num_errors
        self.error_function = error_function
        self.variable_bounds = variable_bounds
        self.variables = [key for key in variable_bounds.keys()]
        #self.variables = variables
        self.template = template
        self.solver = solver.strip().upper()
        self.population_size = population_size

        self.working_variables = []
        self.working_objectives = []
        self.archive_variables = []
        self.archive_objectives = []
        self.unevaluated_variables = []

        self.initialize_platypus()
        self.initialize_mobo()


    def initialize_platypus(self):
        # Initialization for Platypus-Opt Solvers
        if self.solver in ['NSGAII','IBEA']:
            self.problem = platypus.Problem(len(self.variables), self.num_errors)
            self.directions = [platypus.Problem.MINIMIZE for error in range(self.num_errors)]
            self.working_population = []

            self.problem.types = [0 for i in range(len(self.variables))]
            for counter, value in enumerate(self.variables):
                if value[0] == '_':
                    self.problem.types[counter] = Integer(self.variable_bounds[value][0], self.variable_bounds[value][1])
                else:
                    self.problem.types[counter] = Real(self.variable_bounds[value][0], self.variable_bounds[value][1])

        if self.solver == 'NSGAII':
            self.algorithm = platypus.NSGAII(self.problem, self.population_size)
            self.algorithm.variator = platypus.config.default_variator(self.problem)

        if self.solver == 'IBEA':
            self.algorithm = myIBEA(self.problem, self.population_size)
            self.algorithm.variator = platypus.config.default_variator(self.problem)

    def initialize_mobo(self):
        if self.solver == 'MOBO':
            pbounds = []
            for key in self.variable_bounds.keys():
                pbounds.append(self.variable_bounds[key])
            pbounds = np.array(pbounds)
            self.algorithm = myMOBO(target = self.error_function, NObj = self.num_errors, pbounds = pbounds)

    def gen_random_population(self, num_initial_samples = None):
        print('GENERATING RANDOM VARIABLES')
        toreturn = []
        if num_initial_samples is None:
            num_initial_samples = self.population_size

        for i in range(num_initial_samples):
            random_config = [np.random.uniform(self.variable_bounds[self.variables[i]][0], self.variable_bounds[self.variables[i]][1]) for i in range(len(self.variables))]
            toreturn.append(random_config)

        return toreturn


    def evaluate_all_unevaluated(self):
        if len(self.unevaluated_variables) < self.population_size:
            self.unevaluated_variables.extend(self.gen_random_population(self.population_size - len(self.unevaluated_variables)))

        for point in self.unevaluated_variables:
            point_dict = dict(zip(self.variables, point))
            evaluated_point = self.error_function(point_dict)
            self.working_variables.append(point)
            self.working_objectives.append(evaluated_point)
            self.archive_variables.append(point)
            self.archive_objectives.append(evaluated_point)



    def gen_new_samples(self, num_new_samples = None):
        if num_new_samples is None:
            num_new_samples = self.population_size

        if self.solver == 'NSGAII':
            for i in range(len(self.working_variables)):
                mysol = platypus.Solution(self.problem)
                mysol.variables = self.working_variables[i]
                mysol.objectives = self.working_objectives[i]
                self.working_population.append(mysol)

            platypus.core.nondominated_sort(self.working_population)
            self.working_population = platypus.core.nondominated_truncate(self.working_population, self.population_size)

            self.working_variables = []
            self.working_objectives = []
            for point in self.working_population:
                self.working_variables.append(point.variables)
                self.working_objectives.append(point.objectives)

            offspring = []
            while len(offspring) < self.population_size:
                parents = self.algorithm.selector.select(self.algorithm.variator.arity, self.working_population)
                offspring.extend(self.algorithm.variator.evolve(parents))

            new_samples = []
            for single_offspring in offspring:
                new_samples.append(single_offspring.variables)

            return new_samples

        if self.solver == 'IBEA':
            for i in range(len(self.working_variables)):
                mysol = platypus.Solution(self.problem)
                mysol.variables = self.working_variables[i]
                mysol.objectives = self.working_objectives[i]
                self.working_population.append(mysol)

            self.algorithm.fitness_evaluator.evaluate(self.working_population)

            while len(self.working_population) > self.population_size:
                self.algorithm.fitness_evaluator.remove(self.working_population, self.algorithm._find_worst(self.working_population))
                print('removing 1 or 2')

            offspring = []
            while len(offspring) < self.population_size:
                parents = self.algorithm.selector.select(self.algorithm.variator.arity, self.working_population)
                offspring.extend(self.algorithm.variator.evolve(parents))

            new_samples = []
            for single_offspring in offspring:
                new_samples.append(single_offspring.variables)

            return new_samples


        if self.solver == 'MOBO':
            q = 0.5
            prob = 0.1

            for pointID in range(len(self.working_variables)):
                self.algorithm.space.add_observation(np.array(self.working_variables[pointID]), np.array(self.working_objectives[pointID]))

            for i in range(self.algorithm.NObj):
                yy = self.algorithm.space.f[:, i]
                self.algorithm.GP[i].fit(self.algorithm.space.x, yy)

            pop, logbook, front = mobopt._NSGA2.NSGAII(self.algorithm.NObj,
                                         self.algorithm._MOBayesianOpt__ObjectiveGP,
                                         self.algorithm.pbounds,
                                         MU=self.population_size*2)

            Population = np.asarray(pop)
            IndexF, FatorF = self.algorithm._MOBayesianOpt__LargestOfLeast(front, self.algorithm.space.f)
            IndexPop, FatorPop = self.algorithm._MOBayesianOpt__LargestOfLeast(Population,
                                                       self.algorithm.space.x)

            Fator = q * FatorF + (1-q) * FatorPop

            sorted_ids = np.argsort(Fator)

            unevaluated = []

            for i in range(self.population_size):

                Index_try = int(np.argwhere(sorted_ids == np.max(sorted_ids)-i))

                self.algorithm.x_try = Population[Index_try]

                if self.algorithm.space.RS.uniform() < prob:

                    if self.algorithm.NParam > 1:
                        ii = self.algorithm.space.RS.randint(low=0, high=self.algorithm.NParam - 1)
                    else:
                        ii = 0

                    self.algorithm.x_try[ii] = self.algorithm.space.RS.uniform(
                        low=self.algorithm.pbounds[ii][0],
                        high=self.algorithm.pbounds[ii][1])

                unevaluated.append(self.algorithm.x_try)

            return unevaluated



# class OptProblem(Problem):
#     """
#     Class for Forcefield optimization problem. Derived from the generic Platypus Problem class for optimization problems
#     """
#     def __init__(self, num_errors=None, error_function=None, variable_bounds=None, template=None, solver=None, population_size=None):
#         """
#         :param num_errors: Number of errors to be minimized for forcefield optimization
#         :type num_errors: int

#         :param error_function: User-defined function that takes-in a dictionary of variable-value pairs and outputs a list of computed errors
#         :type error_function: function

#         :param variable_bounds: Dictionary of bounds for decision variables in the format `variable: [lower_bound upper_bound]`
#         :type variable_bounds: dict

#         :param template: Forcefield template
#         :type template: str
#         """
#         variables = [key for key in variable_bounds.keys()]
#         super(OptProblem, self).__init__(len(variables), num_errors)
#         for counter, value in enumerate(variables):
#             if value[0] == '_':
#                 self.types[counter] = Integer(variable_bounds[value][0], variable_bounds[value][1])
#             else:
#                 self.types[counter] = Real(variable_bounds[value][0], variable_bounds[value][1])

#         self.error_function = error_function
#         self.directions = [Problem.MINIMIZE for error in range(num_errors)]
#         self.variable_bounds = variable_bounds
#         self.variables = variables
#         self.template = template
#         self.solver = solver
#         self.population_size = population_size

#         self.working_variables = []
#         self.working_objectives = []
#         self.archive_variables = []
#         self.archive_objectives = []



#     def gen_init_population(self, num_initial_samples = None):
#         if num_initial_samples is None:
#             num_initial_samples = self.population_size

#         for i in range(num_initial_samples):
#             random_config = [np.random.uniform(self.variable_bounds[self.variables[i]][0], self.variable_bounds[self.variables[i]][1]) for i in range(len(self.variables))]
#             self.working_variables.append(random_config)




#     def evaluate(self, solution):
#         """
#         Wrapper for platypus.Problem.evaluate . Takes the array of current decision variables and repackages it as a dictionary for the error function

#         :param solution: 1-D array of variables for current epoch and rank
#         :type solution: list
#         """
#         current_var_dict = dict(zip(self.variables, solution.variables))
#         solution.objectives[:] = self.error_function(current_var_dict)





# class Pool(MPIPool):
#     """
#     Wrapper for platypus.MPIPool

#     :param MPIPool: MPI Pool
#     :type MPIPool: MPIPool
#     """
#     def __init__(self, comm=None, debug=False, loadbalance=False):
#         super(Pool, self).__init__(comm=comm, debug=debug, loadbalance=loadbalance)



# def optimize(problem, algorithm, iterations=100, write_forcefields=None):
#     """
#     The optimize function provides a uniform wrapper to solve the EZFF problem using the algorithm(s) provided.

#     :param problem: EZFF Problem to be optimized
#     :type problem: Problem

#     :param algorithm: EZFF Algorithm(s) to use for optimization. Allowed options are ``NSGAII``, ``NSGAIII`` and ``IBEA``, or a list containing any sequence of these options. The algorithms will be used in the sequence provided
#     :type algorithm: str or list (of strings)

#     :param iterations: Number of epochs to perform the optimization for. If multiple algorithms are specified, one iteration value should be provided for each algorithm
#     :type iterations: int or list (of ints)

#     :param write_forcefields: All non-dominated forcefields are written out every ``write_forcefields`` epochs. If this is ``None``, the forcefields are written out for the first and last epoch
#     :type write_forcefields: int or None

#     """
#     # Convert algorithm and iterations into lists
#     if not isinstance(algorithm, list):
#         algorithm = [algorithm]
#     if not isinstance(iterations, list):
#         iterations = [iterations]

#     if not len(algorithm) == len(iterations):
#         raise ValueError("Please provide a maximum number of epochs for each algorithm")

#     total_epochs = 0
#     current_solutions = None
#     for stage in range(0,len(algorithm)):

#         # Construct an algorithm
#         algorithm_for_this_stage = _generate_algorithm(algorithm[stage]["myproblem"],
#                                                       algorithm[stage]["algorithm_string"],
#                                                       algorithm[stage]["population"],
#                                                       algorithm[stage]["mutation_probability"],
#                                                       current_solutions,
#                                                       algorithm[stage]["pool"])

#         if not isinstance(write_forcefields, int):
#             write_forcefields = np.sum([iterations[stage_no] for stage_no in range(stage+1)])

#         for i in range(0, iterations[stage]):
#             total_epochs += 1
#             print('Epoch: '+ str(total_epochs))
#             algorithm_for_this_stage.step()

#             # Make output files/directories
#             outdir = 'results/' + str(total_epochs)
#             if not os.path.isdir(outdir):
#                 os.makedirs(outdir)

#             varfilename = outdir + '/variables'
#             objfilename = outdir + '/errors'
#             varfile = open(varfilename, 'w')
#             objfile = open(objfilename, 'w')
#             for solution in unique(nondominated(algorithm_for_this_stage.result)):
#                 varfile.write(' '.join([str(variables) for variables in solution.variables]))
#                 varfile.write('\n')
#                 objfile.write(' '.join([str(objective) for objective in solution.objectives]))
#                 objfile.write('\n')
#             varfile.close()
#             objfile.close()

#             if total_epochs % write_forcefields == 0:
#                 if not os.path.isdir(outdir+'/forcefields'):
#                     os.makedirs(outdir+'/forcefields')
#                 for sol_index, solution in enumerate(unique(nondominated(algorithm_for_this_stage.result))):
#                     ff_name = outdir + '/forcefields/FF_' + str(sol_index+1)
#                     parameters_dict = dict(zip(problem.variables, solution.variables))
#                     generate_forcefield(problem.template, parameters_dict, outfile=ff_name)

#             current_solutions = algorithm_for_this_stage.population



# def Algorithm(myproblem, algorithm_string, population=1024, mutation_probability=None, pool=None):
#     """
#     Provide a uniform interface to initialize an algorithm class for serial and parallel execution

#     :param myproblem: EZFF Problem to be optimized
#     :type myproblem: Problem

#     :param algorithm_string: EZFF Algorithm to use for optimization. Allowed options are ``NSGAII``, ``NSGAIII`` and ``IBEA``
#     :type algorithm_string: str

#     :param population: Population size for genetic algorithms
#     :type population: int

#     :param mutation_probability: Probability of a decision variable to undergo mutation to a random value within defined bounds
#     :type mutation_probability: float between 0.0 and 1.0

#     :param pool: MPI pool for parallel execution. If this is None, serial execution is assumed
#     :type pool: MPIPool or None
#     """
#     return {"myproblem": myproblem, "algorithm_string": algorithm_string, "population": population, "mutation_probability": mutation_probability, "pool": pool}



# def _pick_algorithm(myproblem, algorithm, population, mutation_probability, current_solution, evaluator=None):
#     """
#     Return a serial or parallel platypus.Algorithm object based on input string

#     :param myproblem: EZFF Problem to be optimized
#     :type myproblem: Problem

#     :param algorithm: EZFF Algorithm to use for optimization. Allowed options are ``NSGAII``, ``NSGAIII`` and ``IBEA``
#     :type algorithm: str

#     :param population: Population size for genetic algorithms
#     :type population: int

#     :param evaluator: Platypus.Evaluator in case of parallel execution
#     :type evaluator: Platypus.Evaluator
#     """

#     if mutation_probability is None:
#         variator = GAOperator(SBX(), PM())
#     else:
#         variator = GAOperator(SBX(), PM(probability=mutation_probability))
#         print('Using provided mutation probability')

#     if algorithm.lower().upper() == 'NSGAII':
#         if evaluator is None:
#             if current_solution is None:
#                 return NSGAII(myproblem, population_size=population, variator=variator)
#             else:
#                 return NSGAII(myproblem, population_size=population, variator=variator, generator=InjectedPopulation(current_solution[0:population]))
#         else:
#             if current_solution is None:
#                 return NSGAII(myproblem, population_size=population, variator=variator, evaluator=evaluator)
#             else:
#                 return NSGAII(myproblem, population_size=population, variator=variator, generator=InjectedPopulation(current_solution[0:population]), evaluator=evaluator)
#     elif algorithm.lower().upper() == 'NSGAIII':
#         num_errors = len(myproblem.directions)
#         divisions = int(np.power(population * np.math.factorial(num_errors-1), 1.0/(num_errors-1))) + 2 - num_errors
#         divisions = np.maximum(1, divisions)
#         if evaluator is None:
#             if current_solution is None:
#                 return NSGAIII(myproblem, divisions_outer=divisions, variator=variator)
#             else:
#                 return NSGAIII(myproblem, divisions_outer=divisions, variator=variator, generator=InjectedPopulation(current_solution[0:population]))
#         else:
#             if current_solution is None:
#                 return NSGAIII(myproblem, divisions_outer=divisions, variator=variator, evaluator=evaluator)
#             else:
#                 return NSGAIII(myproblem, divisions_outer=divisions, variator=variator, generator=InjectedPopulation(current_solution[0:population]), evaluator=evaluator)
#     elif algorithm.lower().upper() == 'IBEA':
#         if evaluator is None:
#             if current_solution is None:
#                 return IBEA(myproblem, population_size=population, variator=variator)
#             else:
#                 return IBEA(myproblem, population_size=population, variator=variator, generator=InjectedPopulation(current_solution[0:population]))
#         else:
#             if current_solution is None:
#                 return IBEA(myproblem, population_size=population, variator=variator, evaluator=evaluator)
#             else:
#                 return IBEA(myproblem, population_size=population, variator=variator, generator=InjectedPopulation(current_solution[0:population]), evaluator=evaluator)
#     else:
#         raise Exception('Please enter an algorithm for optimization. NSGAII , NSGAIII , IBEA are supported')



# def _generate_algorithm(myproblem, algorithm_string, population, mutation_probability, current_solution, pool):
#     if pool is None:
#         algorithm = _pick_algorithm(myproblem, algorithm_string, population=population, mutation_probability=mutation_probability, current_solution=current_solution)
#     else:
#         if not pool.is_master():
#             pool.wait()
#             sys.exit(0)
#         else:
#             evaluator = PoolEvaluator(pool)
#             algorithm = _pick_algorithm(myproblem, algorithm_string, population=population, mutation_probability=mutation_probability, current_solution=current_solution, evaluator=evaluator)
#     return algorithm
