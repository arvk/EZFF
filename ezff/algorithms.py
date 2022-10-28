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
