Examples
========
The following forcefield optimization examples showcase the features of EZFF


lj-gulp-serial
------------
Optimization of a Lennard-Jones forcefield for FCC Neon against 2 objectives -- **bulk modulus** and **lattice constant**

Features demonstrated in this example

1. Basic use of forcefield templates and variable_range files
2. Reading-in elastic modulus tensor from GULP run
3. Reading-in lattice constants after a GULP run


lj-lammps-serial
------------
Optimization of a Lennard-Jones forcefield for FCC Neon against 2 objectives -- **bulk modulus** and **lattice constant**

Features demonstrated in this example

1. Reading-in elastic modulus tensor from LAMMPS run
2. Reading-in lattice constants after a LAMMPS run


sw-gulp-serial
------------
Optimization of a Stillinger-Weber forcefield for the 2H-MoSe2 monolayer system against 3 objectives -- **Lattice constant** (*a*), **Elastic modulus** (:math:`C_{11}`) and **Phonon dispersion**

Features demonstrated in this example

1. Reading-in phonon dispersion from GULP and VASP data files
2. Calculating error between phonon dispersions
3. Calculating error between computed and ground-truth phonon dispersions
4. Reading-in elastic modulus tensor from GULP run
5. Usage of the Multi-objective Bayesian Optimizer


sw-gulp-multialgo
--------------
Parallel optimization of a Stillinger-Weber forcefield for the 1T' monolayer system against 6 objectives -- **Two lattice constants** (*a* and *b*), **One elastic modulus** (:math:`C_{11}`) and **Three phonon dispersion curves** (one each for compressed, relaxed and expanded crystals) using a sequence of multiple multi-objective genetic algorithms. Here, the population from the last epoch of optimization with a single algorithm is used as the initial population for the next algorithm in the sequence.

Features demonstrated in this example

1. Using multiple genetic algorithms in sequence for a single problem
2. Use of different population sizes and epochs for each algorithm in the sequence


sw-gulp-parallel-multi
--------------
Parallel optimization of a Stillinger-Weber forcefield for the 1T' monolayer system against 6 objectives -- **Two lattice constants** (*a* and *b*), **One elastic modulus** (:math:`C_{11}`) and **Three phonon dispersion curves** (one each for compressed, relaxed and expanded crystals)

Features demonstrated in this example

1. Spawning and using Multiprocessing pools for optimization
2. Non-uniform weighting schemes for calculating phonon dispersion errors


sw-gulp-parallel-mpi
--------------
Parallel optimization of a Stillinger-Weber forcefield for the 1T' monolayer system against 6 objectives -- **Two lattice constants** (*a* and *b*), **One elastic modulus** (:math:`C_{11}`) and **Three phonon dispersion curves** (one each for compressed, relaxed and expanded crystals)

Features demonstrated in this example

1. Spawning and using MPI pools for optimization


vashishta-lammps-serial
------------
Optimization of a Stillinger-Weber forcefield for SiC crystal against 2 objectives -- **Lattice constant** (*a*) and **Elastic modulus** (:math:`C_{11}`)

Features demonstrated in this example

1. Reading-in elastic modulus tensor from LAMMPS run
2. Optimization of Vashishta potential


reaxff-charge-gulp-serial
------------------
Optimization of charge parameters in the ReaxFF forcefield for a thio-ketone monomer against 1 objective -- **atomic charges**

Features demonstrated in this example

1. Use of make_template_qeq
2. Use of ezff.error_atomic_charges
3. Use of nevergrad single-objective optimizers


reaxff-distortion-gulp-serial
------------------
Optimization of charge and bond parameters in the ReaxFF forcefield for a thio-ketone monomer against 2 objective -- **atomic charges** and **structural distortion**

Features demonstrated in this example

1. Use of ezff.error_structure_distortion



reaxff-lammps-parallel-multi
--------------
Parallel optimization of ReaxFF forcefield for a thio-ketone monomer against 2 objectives -- **Dissociation energy** of the C-S bond and C-S **vibrational frequency**

Features demonstrated in this example

1. Using QChem interface to read-in QM energies
2. Using LAMMPS interface to perform single-point calculations and read-in energy
3. Using utils.reaxff methods for generating forcefields templates and variable range files
4. Heterogeneous weighting scheme for calculating errors from potential energy surface scans


lj-gulp-save-restart
-------------------
Serial optimization of Lennard Jones forcefield for solid Neon against 2 objectives -- **Lattice constant** (*a*) and **Elastic modulus** (:math:`C_{11}`)

Features demonstrated in this example

1. Save evaluated variables as numpy arrays
2. Continue optimization after loading pre-evaluated variables


pedone-lammps-parallel-multi
-------------------
Serial optimization of the Pedone forcefield (hybrid mixture of Coulombic + Morse + Repulsive interactions) for amorphous SiO2 against 2 objectives -- **Lattice constant** (*a*) and **Elastic modulus** (:math:`C_{11}`)

Features demonstrated in this example

1. Parameterization of hybrid forcefields (containing 2 or more forcefield types) in LAMMPS
