Algorithms
===========
EZFF comes with several algorithms for gradient-free single- and multi-objective parameterization of forcefields. Algorithms are provided from one of four optimization frameworks - Nevergrad, Platypus, PyMOO, and MOBOpt. The following algorithms are available in EZFF v1.0.

.. list-table:: Available EZFF Algorithms
   :widths: 25 40 15 20
   :header-rows: 1

   * - Algorithm name
     - Algorithm type
     - Framework
     - Number of Objecives
   * - `ngopt_so`
     - Adaptible meta-optimizer
     - Nevergrad
     - Single
   * - `twopointsde_so`
     - Differential Evolution with 2-points crossover
     - Nevergrad
     - Single
   * - `portfoliodiscreteoneplusone_so`
     - Genetic Algorithm for mixed discrete/continuous search spaces
     - Nevergrad
     - Single
   * - `oneplusone_so`
     - One Plus One
     - Nevergrad
     - Single
   * - `twopointsde_so`
     - Differential Evolution with 2-points crossover
     - Nevergrad
     - Single
   * - `cma_so`
     - Covariance Matrix Adaptation Evolution Strategy
     - Nevergrad
     - Single
   * - `tbpsa_so`
     - Test-based population size adaptation
     - Nevergrad
     - Single
   * - `pso_so`
     - Particle Swarm Optimization
     - Nevergrad
     - Single
   * - `scrhammersleysearchplusmiddlepoint_so`
     - Scrambled-Hammersley plus middle point single-shot optimization
     - Nevergrad
     - Single
   * - `randomsearch_so`
     - Random sampling
     - Nevergrad
     - Single
   * - `nsga2_mo_pymoo`
     - Nondominated Sorting Genetic Algorithm II
     - pymoo
     - Multiple
   * - `nsga3_mo_pymoo`
     - Nondominated Sorting Genetic Algorithm III
     - pymoo
     - Multiple
   * - `unsga3_mo_pymoo`
     - Unified Nondominated Sorting Genetic Algorithm III with tournament pressure
     - pymoo
     - Multiple
   * - `ctaea_mo_pymoo`
     - Constrained Two-Archive Evolutionary Algorithm
     - pymoo
     - Multiple
   * - `smsemoa_mo_pymoo`
     - S-Metric Selection Evolutionary Multiobjective Optimization Algorithm
     - pymoo
     - Multiple
   * - `rvea_mo_pymoo`
     - Reference Vector Guided Evolutionary Algorithm
     - pymoo
     - Multiple
   * - `es_so_pymoo`
     - Evolutionary Strategy
     - pymoo
     - Single
   * - `neldermead_so_pymoo`
     - Nelder Mead
     - pymoo
     - Single
   * - `cmaes_so_pymoo`
     - Covariance Matrix Adaptation Evolution Strategy
     - pymoo
     - Single
   * - `nsga2_mo_platypus`
     - Nondominated Sorting Genetic Algorithm II
     - Platypus
     - Multiple
   * - `nsga3_mo_platypus`
     - Nondominated Sorting Genetic Algorithm III
     - Platypus
     - Multiple
   * - `gde3_mo_platypus`
     - Generalized Differential Evolution 3
     - Platypus
     - Multiple
   * - `mobo`
     - Multi-objective Bayesian Optimization
     - MOBOpt
     - Multiple
