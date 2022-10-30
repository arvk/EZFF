"""This module provide general functions for EZFF"""
import os
import sys
import xtal
import numpy as np
import platypus
from platypus import Problem, unique, nondominated, NSGAII, NSGAIII, IBEA, PoolEvaluator, Solution
from platypus.types import Real, Integer
from platypus.operators import InjectedPopulation, GAOperator, SBX, PM, TournamentSelector, RandomGenerator
from platypus.config import default_variator
from mobopt import MOBayesianOpt



class myIBEA(IBEA):
    def __init__(self, problem, population_size = 100):
        super(myIBEA, self).__init__(problem, population_size)

    def _find_worst(self, population):
        index = 0
        for i in range(1, len(population)):
            if self.fitness_comparator.compare(population[index], population[i]) < 0:
                index = i
        return index



class myN2(NSGAII):
    def __init__(self, problem,
                 population_size = 300,
                 generator = RandomGenerator(),
                 selector = TournamentSelector(2),
                 variator = None,
                 archive = None,
                 **kwargs):
        super(myN2, self).__init__(problem, population_size, generator, **kwargs)
        self.selector = selector
        self.variator = default_variator(problem)
        self.archive = archive
        self.population_size = population_size

    def gen_start_pop_default(self):
        self.population = [self.generator.generate(self.problem) for _ in range(self.population_size)]

    def gen_start_pop_dictonly(self):
        self.population = [self.generator.generate(self.problem).variables for _ in range(self.population_size)]

    def gen_start_pop_fromdict(self):
        self.population = []
        varonly = [self.generator.generate(self.problem).variables for _ in range(self.population_size)]
        for var in varonly:
            mysol = Solution(self.problem)
            mysol.variables = var
            self.population.append(mysol)

    measured_variables = []
    measured_objectives = []
    all_measured_variables = []
    all_measured_objectives = []



class myMOBO(MOBayesianOpt):
    def __init__(self, target, NObj, pbounds, constraints=[],
                 verbose=False, Picture=False, TPF=None,
                 n_restarts_optimizer=10, Filename=None,
                 MetricsPS=True, max_or_min='max', RandomSeed=None,
                 kernel=None):
        super(myMOBO, self).__init__(target, NObj, pbounds, constraints=[],
                 verbose=False, Picture=False, TPF=None,
                 n_restarts_optimizer=10, Filename=None,
                 MetricsPS=True, max_or_min='max', RandomSeed=None,
                 kernel=None)

    def mymax1(self,
             n_iter=100,
             prob=0.1,
             ReduceProb=False,
             q=0.5,
             n_pts=100,
             SaveInterval=10,
             FrontSampling=[10, 25, 50, 100]):

        self.q = q
        self.NewProb = prob

        for i in range(self.NObj):
            yy = self.space.f[:, i]
            self.GP[i].fit(self.space.x, yy)


        pop, logbook, front = mobopt._NSGA2.NSGAII(self.NObj,
                                     self._MOBayesianOpt__ObjectiveGP,
                                     self.pbounds,
                                     MU=n_pts*2)

        Population = np.asarray(pop)
        IndexF, FatorF = self._MOBayesianOpt__LargestOfLeast(front, self.space.f)
        IndexPop, FatorPop = self._MOBayesianOpt__LargestOfLeast(Population,
                                                   self.space.x)

        Fator = self.q * FatorF + (1-self.q) * FatorPop

        sorted_ids = np.argsort(Fator)

        unevaluated = []

        for i in range(n_pts):

            Index_try = int(np.argwhere(sorted_ids == np.max(sorted_ids)-i))

            self.x_try = Population[Index_try]

            if self.space.RS.uniform() < self.NewProb:

                if self.NParam > 1:
                    ii = self.space.RS.randint(low=0, high=self.NParam - 1)
                else:
                    ii = 0

                self.x_try[ii] = self.space.RS.uniform(
                    low=self.pbounds[ii][0],
                    high=self.pbounds[ii][1])

            unevaluated.append(self.x_try)

        return unevaluated
