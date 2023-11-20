"""This module provide general functions for EZFF"""

# General imports
import os
import sys
import numpy as np
import math
import random
from datetime import datetime
import xtal
from functools import partial

# EZFF imports
from .ffio import *
from .errors import *

# Optimizer imports
import nevergrad as ng
import mobopt
from pymoo.core.problem import Problem as pymoo_Problem
from pymoo.core.individual import Individual as pymoo_Individual
from pymoo.core.population import Population as pymoo_Population
from pymoo.algorithms.moo.nsga2 import NSGA2 as pymoo_NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3 as pymoo_NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3 as pymoo_UNSGA3
from pymoo.algorithms.moo.ctaea import CTAEA as pymoo_CTAEA
from pymoo.algorithms.moo.sms import SMSEMOA as pymoo_SMSEMOA
from pymoo.util.ref_dirs import get_reference_directions as pymoo_get_reference_directions
from pymoo.algorithms.moo.rvea import RVEA as pymoo_RVEA
from pymoo.termination.max_eval import MaximumFunctionCallTermination
from pymoo.algorithms.soo.nonconvex.es import ES as pymoo_ES
from pymoo.algorithms.soo.nonconvex.nelder import NelderMead as pymoo_NelderMead
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES as pymoo_CMAES
import platypus

# Parallel processing imports
from schwimmbad import MultiPool as sch_MultiPool
from schwimmbad import MPIPool as sch_MPIPool
from mpi4py import MPI
import multiprocessing



__version__ = '1.0.2' # Update setup.py if version changes


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
        self.total_epochs = 0

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
        pymoo_algos = ['NSGA2_MO_PYMOO', 'NSGA3_MO_PYMOO', 'UNSGA3_MO_PYMOO', 'CTAEA_MO_PYMOO', 'SMSEMOA_MO_PYMOO', 'RVEA_MO_PYMOO', 'ES_SO_PYMOO', 'NELDERMEAD_SO_PYMOO', 'CMAES_SO_PYMOO']
        platypus_algos = ['NSGA2_MO_PLATYPUS', 'NSGA3_MO_PLATYPUS', 'GDE3_MO_PLATYPUS']

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

        elif algo_string.upper() in pymoo_algos:
            self.algo_framework = 'pymoo'
            var_mins = []
            var_maxs = []
            for variable in self.variable_bounds.keys():
                var_mins.append(self.variable_bounds[variable][0])
                var_maxs.append(self.variable_bounds[variable][1])
            self.pymoo_problem = pymoo_Problem(n_var = self.num_variables, n_obj = self.num_errors, n_constr = 0, xl = var_mins, xu = var_maxs)

            self.normalize_errors()
            initial_population = []
            for varid, var in enumerate(self.variables):
                evaledsoln = pymoo_Individual()
                evaledsoln._X = var
                evaledsoln._F = self.normalized_errors[varid]
                initial_population.append(evaledsoln)

            if algo_string.upper() == 'NSGA2_MO_PYMOO':
                if initial_population == []:
                    self.algorithm = pymoo_NSGA2(self.population_size)
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_NSGA2(self.population_size, sampling = initial_population)
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

            elif algo_string.upper() == 'NSGA3_MO_PYMOO':
                # Identify number of reference points
                min_points = [math.comb(self.num_errors + ref_pts - 1, ref_pts) for ref_pts in range(25)]
                num_reference_points = np.sum(np.array(min_points) < self.population_size)
                reference_directions = pymoo_get_reference_directions("das-dennis", self.num_errors, n_partitions=num_reference_points)
                if initial_population == []:
                    self.algorithm = pymoo_NSGA3(pop_size=self.population_size, ref_dirs=reference_directions)
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_NSGA3(pop_size=self.population_size, ref_dirs=reference_directions, sampling = initial_population)
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

            elif algo_string.upper() == 'UNSGA3_MO_PYMOO':
                # Identify number of reference points
                min_points = [math.comb(self.num_errors + ref_pts - 1, ref_pts) for ref_pts in range(25)]
                num_reference_points = np.sum(np.array(min_points) < self.population_size)
                reference_directions = pymoo_get_reference_directions("das-dennis", self.num_errors, n_partitions=num_reference_points)
                if initial_population == []:
                    self.algorithm = pymoo_UNSGA3(pop_size=self.population_size, ref_dirs=reference_directions)
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_UNSGA3(pop_size=self.population_size, ref_dirs=reference_directions, sampling = initial_population)
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

            elif algo_string.upper() == 'CTAEA_MO_PYMOO':
                # Identify number of reference points
                min_points = [math.comb(self.num_errors + ref_pts - 1, ref_pts) for ref_pts in range(25)]
                num_reference_points = np.sum(np.array(min_points) < self.population_size)
                reference_directions = pymoo_get_reference_directions("das-dennis", self.num_errors, n_partitions=num_reference_points)
                if initial_population == []:
                    self.algorithm = pymoo_CTAEA(ref_dirs=reference_directions)
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_CTAEA(ref_dirs=reference_directions, sampling = initial_population)
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

            elif algo_string.upper() == 'SMSEMOA_MO_PYMOO':
                if initial_population == []:
                    self.algorithm = pymoo_SMSEMOA(self.population_size)
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_SMSEMOA(self.population_size, sampling = initial_population)
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

            elif algo_string.upper() == 'RVEA_MO_PYMOO':
                # Identify number of reference points
                min_points = [math.comb(self.num_errors + ref_pts - 1, ref_pts) for ref_pts in range(25)]
                num_reference_points = np.sum(np.array(min_points) < self.population_size)
                reference_directions = pymoo_get_reference_directions("das-dennis", self.num_errors, n_partitions=num_reference_points)
                if initial_population == []:
                    self.algorithm = pymoo_RVEA(ref_dirs=reference_directions, pop_size = self.population_size)
                    termination = MaximumFunctionCallTermination(5000)
                    self.algorithm.termination = termination
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_RVEA(ref_dirs=reference_directions, sampling = initial_population, pop_size = self.population_size)
                    termination = MaximumFunctionCallTermination(5000)
                    self.algorithm.termination = termination
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

            elif algo_string.upper() == 'ES_SO_PYMOO':
                if initial_population == []:
                    self.algorithm = pymoo_ES(n_offspring = self.population_size, pop_size = self.population_size)
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_ES(n_offspring = self.population_size, pop_size = self.population_size, sampling = initial_population)
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

            elif algo_string.upper() == 'NELDERMEAD_SO_PYMOO':
                if initial_population == []:
                    self.algorithm = pymoo_NelderMead()
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_NelderMead()
                    self.algorithm.pop = initial_population
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

            elif algo_string.upper() == 'CMAES_SO_PYMOO':
                x0_search = np.mean(self.variables, axis=0)
                if initial_population == []:
                    self.algorithm = pymoo_CMAES(x0 = x0_search)
                else:
                    initial_population = pymoo_Population(initial_population)
                    self.algorithm = pymoo_CMAES(x0 = x0_search)
                    self.algorithm.pop = initial_population
                self.algorithm.setup(self.pymoo_problem, seed = np.random.randint(100000), verbose = False)

        elif algo_string.upper() in platypus_algos:
            self.algo_framework = 'platypus'
            var_mins = []
            var_maxs = []
            for variable in self.variable_bounds.keys():
                var_mins.append(self.variable_bounds[variable][0])
                var_maxs.append(self.variable_bounds[variable][1])
            self.platypus_problem = platypus.Problem(self.num_variables, self.num_errors)
            self.platypus_problem.types = [platypus.Real(var_mins[i], var_maxs[i]) for i in range(self.num_variables)]

            self.normalize_errors()
            initial_population = []
            for varid, var in enumerate(self.variables):
                evaledsoln = platypus.Solution(self.platypus_problem)
                evaledsoln.variables[:] = list(var)
                evaledsoln.objectives[:] = list(self.normalized_errors[varid])
                initial_population.append(evaledsoln)

            if algo_string.upper() == 'NSGA2_MO_PLATYPUS':
                self.algorithm = platypus.NSGAII(self.platypus_problem, self.population_size)
                self.algorithm.variator = platypus.default_variator(self.platypus_problem)
                if len(initial_population) > 0:
                    self.algorithm.population = initial_population

            elif algo_string.upper() == 'NSGA3_MO_PLATYPUS':
                # Identify number of reference points
                min_points = [math.comb(self.num_errors + ref_pts - 1, ref_pts) for ref_pts in range(200)]
                num_reference_points = np.sum(np.array(min_points) < self.population_size)
                self.algorithm = platypus.NSGAIII(self.platypus_problem, divisions_outer = num_reference_points)
                self.algorithm.variator = platypus.default_variator(self.platypus_problem)
                if len(initial_population) > 0:
                    self.algorithm.population = initial_population

            if algo_string.upper() == 'GDE3_MO_PLATYPUS':
                self.algorithm = platypus.GDE3(self.platypus_problem, population_size = self.population_size)
                if len(initial_population) > 0:
                    self.algorithm.population = initial_population



    def ask(self):
        """
        Ask the optimization algorithm the next candidate variables for evaluation
        """
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
                self.algorithm.space.add_observation(np.array(self.variables[pointID]), np.array(self.normalized_errors[pointID]))

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


        elif self.algo_framework == 'pymoo':
            while (len(new_variables) < self.population_size):
                newXs = self.algorithm.ask()

                for newX in newXs:
                    new_variables.append(newX.X)

            return new_variables


        elif self.algo_framework == 'platypus':
            if self.algo_string.upper() == 'NSGA2_MO_PLATYPUS':
                platypus.nondominated_sort(self.algorithm.population)
                self.algorithm.population = platypus.nondominated_truncate(self.algorithm.population, self.population_size)
                for i in range(self.population_size):
                    parents = self.algorithm.selector.select(self.algorithm.variator.arity, self.algorithm.population)
                    single_offspring = self.algorithm.variator.evolve(parents)
                    new_variables.append(single_offspring[0].variables[:])
                    new_variables.append(single_offspring[1].variables[:])
                new_variables = random.sample(new_variables, self.population_size)
            elif self.algo_string.upper() == 'NSGA3_MO_PLATYPUS':
                platypus.nondominated_sort(self.algorithm.population)
                self.algorithm.population = self.algorithm._reference_point_truncate(self.algorithm.population, self.population_size)
                for i in range(self.population_size):
                    parents = self.algorithm.selector.select(self.algorithm.variator.arity, self.algorithm.population)
                    single_offspring = self.algorithm.variator.evolve(parents)
                    new_variables.append(single_offspring[0].variables[:])
                    new_variables.append(single_offspring[1].variables[:])
                new_variables = random.sample(new_variables, self.population_size)
            elif self.algo_string.upper() == 'GDE3_MO_PLATYPUS':
                self.algorithm.population = self.algorithm.survival(self.algorithm.population)
                for i in range(self.population_size):
                    parents = self.algorithm.select(i, self.algorithm.variator.arity)
                    single_offspring = self.algorithm.variator.evolve(parents)
                    new_variables.append(single_offspring[0].variables[:])
                #     new_variables.append(single_offspring[1].variables[:])
                # new_variables = random.sample(list(set(new_variables)), self.population_size)

            return new_variables



    def parameterize(self, num_epochs = None, pool = None):
        """
        The optimize function provides a uniform wrapper to solve the EZFF problem using the algorithm(s) provided.

        :param num_epochs: Number of epochs to perform the optimization for. If multiple algorithms are specified, one iteration value should be provided for each algorithm
        :type num_epochs: int

        :param pool: Multiprocessing or MPI Pool for forcefield parameterization
        :type pool: Multiprocessing or MPI Pool object
        """
        self.pool = pool
        if self.algo_framework == 'nevergrad':
            for epoch in range(num_epochs):

                self.total_epochs += 1
                self.normalize_errors()

                self.set_algorithm(algo_string = self.algo_string, population_size = self.population_size)

                for computed_id, computed in enumerate(self.variables):
                    computed_value = {k:v for (k,v) in zip(self.variable_names,computed)}
                    self.algorithm.suggest(computed_value)
                    asked_suggestion = self.algorithm.ask()
                    self.algorithm.tell(asked_suggestion, self.normalized_errors[computed_id])

                new_variables = self.ask()
                new_errors = []

                if self.pool is None:
                    for variable in new_variables:
                        variable_dict = dict(zip(self.variable_names, variable))
                        error = self.error_function(variable_dict, self.forcefield_template)
                        new_errors.append(error)
                else:
                    if self.pool_type == 'mpi':
                        if not self.pool.is_master():
                            pool.wait()

                    variable_dict_list = [dict(zip(self.variable_names, variable)) for variable in new_variables]
                    new_errors = self.pool.map(partial(self.error_function, template = self.forcefield_template), variable_dict_list)

                for variable_id, variable in enumerate(new_variables):
                    self.variables.append(variable)
                    self.errors.append(new_errors[variable_id])

                self._write_out_forcefields()


        elif self.algo_framework == 'mobopt':
            for epoch in range(num_epochs):

                self.total_epochs += 1
                self.normalize_errors()

                self.set_algorithm(algo_string = self.algo_string, population_size = self.population_size)

                new_variables = self.ask()
                new_errors = []

                if self.pool is None:
                    for variable in new_variables:
                        variable_dict = dict(zip(self.variable_names, variable))
                        error = self.error_function(variable_dict, self.forcefield_template)
                        new_errors.append(error)
                else:
                    if self.pool_type == 'mpi':
                        if not self.pool.is_master():
                            pool.wait()

                    variable_dict_list = [dict(zip(self.variable_names, variable)) for variable in new_variables]
                    new_errors = self.pool.map(partial(self.error_function, template = self.forcefield_template), variable_dict_list)

                for variable_id, variable in enumerate(new_variables):
                    self.variables.append(variable)
                    self.errors.append(new_errors[variable_id])

                self._write_out_forcefields()


        elif self.algo_framework == 'pymoo':
            for epoch in range(num_epochs):

                self.total_epochs += 1
                self.normalize_errors()

                self.set_algorithm(algo_string = self.algo_string, population_size = self.population_size)

                new_variables = self.ask()
                new_errors = []

                if self.pool is None:
                    for variable in new_variables:
                        variable_dict = dict(zip(self.variable_names, variable))
                        error = self.error_function(variable_dict, self.forcefield_template)
                        new_errors.append(error)
                else:
                    if self.pool_type == 'mpi':
                        if not self.pool.is_master():
                            pool.wait()

                    variable_dict_list = [dict(zip(self.variable_names, variable)) for variable in new_variables]
                    new_errors = self.pool.map(partial(self.error_function, template = self.forcefield_template), variable_dict_list)

                for variable_id, variable in enumerate(new_variables):
                    self.variables.append(variable)
                    self.errors.append(new_errors[variable_id])

                self._write_out_forcefields()


        elif self.algo_framework == 'platypus':
            for epoch in range(num_epochs):

                self.total_epochs += 1
                self.normalize_errors()

                self.set_algorithm(algo_string = self.algo_string, population_size = self.population_size)

                new_variables = self.ask()
                new_errors = []

                if self.pool is None:
                    for variable in new_variables:
                        variable_dict = dict(zip(self.variable_names, variable))
                        error = self.error_function(variable_dict, self.forcefield_template)
                        new_errors.append(error)

                    # for variable_id, variable in enumerate(new_variables):
                    #     self.variables.append(variable)
                    #     self.errors.append(new_errors[variable_id])

                else:
                    if self.pool_type == 'mpi':
                        if not self.pool.is_master():
                            pool.wait()

                    variable_dict_list = [dict(zip(self.variable_names, variable)) for variable in new_variables]
                    new_errors = self.pool.map(partial(self.error_function, template = self.forcefield_template), variable_dict_list)

                    # for variable_id, variable in enumerate(new_variables):
                    #     self.variables.append(variable)
                    #     self.errors.append(new_errors[variable_id])


                for variable_id, variable in enumerate(new_variables):
                    self.variables.append(variable)
                    self.errors.append(new_errors[variable_id])

                self._write_out_forcefields()



    def _write_out_forcefields(self):
        print('Epoch: '+ str(self.total_epochs))

        if not (self.algo_string.upper() == 'RANDOMSEARCH_SO'):
            # Make output files/directories
            outdir = 'results/' + str(self.total_epochs)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

            varfilename = outdir + '/variables'
            errfilename = outdir + '/errors'
            varfile = open(varfilename, 'w')
            errfile = open(errfilename, 'w')

            reco_vars, reco_errs = self.get_best_recommendation()

            if self.algo_framework == 'nevergrad' and self.num_errors == 1:
                reco_vars = [[reco_vars[key] for key in self.variable_names]]
                reco_errs = [[reco_errs]]

            for reco_id, reco_var in enumerate(reco_vars):
                varfile.write(' '.join([str(variable) for variable in reco_var]))
                varfile.write('\n')
                if self.algo_framework == 'mobopt':
                    errfile.write(' '.join([str(0.0 - error) for error in reco_errs[reco_id]]))
                else:
                    errfile.write(' '.join([str(error) for error in reco_errs[reco_id]]))
                errfile.write('\n')
            varfile.close()
            errfile.close()

            all_evaluated_filename = outdir + '/all_evaluated_'
            self.save_evaluated(all_evaluated_filename)

            if self.total_epochs % 5 == 0:
                if not os.path.isdir(outdir+'/forcefields'):
                    os.makedirs(outdir+'/forcefields')
                for reco_id, reco_var in enumerate(reco_vars):
                #for sol_index, solution in enumerate(unique(nondominated(algorithm_for_this_stage.result))):
                    ff_name = outdir + '/forcefields/FF_' + str(reco_id+1)
                    parameters_dict = dict(zip(self.variable_names, reco_var))
                    generate_forcefield(self.forcefield_template, parameters_dict, outfile=ff_name)






    def get_best_recommendation(self):
        """
        Return the best variables evaluated so far
        """
        best_variables = None
        best_errors = None
        if self.algo_framework == 'nevergrad':
            best_recommendation = self.algorithm.provide_recommendation()
            best_variables = best_recommendation.value
            best_errors = best_recommendation.loss
        elif self.algo_framework == 'mobopt':
            best_errors, best_variables = self.algorithm.space.ParetoSet()
        elif self.algo_framework == 'pymoo':
            self.normalize_errors()
            initial_population = []
            for varid, var in enumerate(self.variables):
                evaledsoln = pymoo_Individual()
                evaledsoln._X = var
                evaledsoln._F = self.normalized_errors[varid]
                initial_population.append(evaledsoln)
            initial_population = pymoo_Population(initial_population)
            if '_SO_' in self.algo_string:
                self.algorithm.pop = initial_population
                self.algorithm._set_optimum()
            else:
                self.algorithm.tell(infills = initial_population)
            best_recommendation = self.algorithm.result()
            best_variables = best_recommendation.X
            best_errors = best_recommendation.F
        elif self.algo_framework == 'platypus':
            platypus.nondominated_sort(self.algorithm.population)
            recommendation = platypus.unique(platypus.nondominated(self.algorithm.population))
            best_variables = []
            best_errors = []
            recommendation = list(set(recommendation))   # Remove duplicates
            for single_reco in recommendation:
                best_variables.append(single_reco.variables[:])
                best_errors.append(single_reco.objectives[:])

        return [best_variables, best_errors]


    def generate_pool(self, pool_type):
        """
        Return a parallel pool object

        :param pool_type: Type of parallel pool
        :type pool_type: str
        """
        if pool_type == 'multi':
            self.pool_type = 'multi'
            return sch_MultiPool()
        elif pool_type == 'mpi':
            self.pool_type = 'mpi'
            return sch_MPIPool()

    def save_evaluated(self,filename):
        """
        Save all variables evaluated so far as a zipped numpy array

        :param filename: File to which variables are saved
        :type filename: str
        """
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = filename+timestamp+'.npz'
        np.savez(filename, variables=np.array(self.variables), errors=np.array(self.errors))

    def load_evaluated(self,filename):
        """
        Load all variables evaluated from a zipped numpy array

        :param filename: File to load variables from
        :type filename: str
        """
        fileobj = np.load(filename)
        vars = fileobj['variables']
        errs = fileobj['errors']
        for varid, var in enumerate(vars):
            self.variables.append(var)
            self.errors.append(errs[varid])


    def normalize_errors(self, scale = 2.0):
        self.normalized_errors = []
        if len(self.errors) > 0:
            maximums = np.nanmax(self.errors, axis=0)
            maximums[np.isnan(maximums)] = 100.0
            for var in self.errors:
                if np.any(np.isnan(var)):
                    self.normalized_errors.append(maximums * scale)
                else:
                    self.normalized_errors.append(var)

        self.normalized_errors = np.array(self.normalized_errors)
        if self.algo_framework == 'mobopt':
            self.normalized_errors = 0.0 - self.normalized_errors    # MOBOPT will only attempt to maximize the error function



def get_pool_rank():
    """
    Return the rank of the current process in a parallel setting
    """
    try:
        myrank = multiprocessing.current_process()._identity[0]
        return myrank                     # return rank of multiprocessing process, if available
    except:
        return MPI.COMM_WORLD.Get_rank()  # return rank of MPI process, or 0 if there is no MPI
