Examples
========
The following forcefield optimization examples showcase the features of EZFF


lj-serial
------------
Optimization of a Lennard-Jones forcefield for FCC Neon against 2 objectives -- **bulk modulus** and **lattice constant**

Features demonstrated in this example

1. Basic use of forcefield templates and variable_range files
2. Reading-in elastic modulus tensor from GULP run
3. Reading-in lattice constants after a GULP run


lj-parallel
------------
Parallel optimization of a Lennard-Jones forcefield for FCC Neon against 2 objectives -- **bulk modulus** and **lattice constant**

Features demonstrated in this example

1. All features from lj-serial
2. Basic use of ezff.Pool for parallel optimization


sw-serial
------------
Optimization of a Stillinger-Weber forcefield for the 2H-MoSe2 monolayer system against 3 objectives -- **Lattice constant** (*a*), **Elastic modulus** (:math:`C_{11}`) and **Phonon dispersion**

Features demonstrated in this example

1. Reading-in phonon dispersion from GULP and VASP data files
2. Calculating error between phonon dispersions
3. Calculating error between computed and ground-truth phonon dispersions
4. Reading-in elastic modulus tensor from GULP run


sw-parallel
--------------
Parallel optimization of a Stillinger-Weber forcefield for the 1T' monolayer system against 6 objectives -- **Two lattice constants** (*a* and *b*), **One elastic modulus** (:math:`C_{11}`) and **Three phonon dispersion curves** (one each for compressed, relaxed and expanded crystals)

Features demonstrated in this example

1. All features from sw-serial
2. Spawning and using MPI pools for optimization
3. Non-uniform weighting schemes for calculating phonon dispersion errors


rxff-serial
--------------
Optimization of ReaxFF forcefield for a thio-ketone monomer against 2 objectives -- **Dissociation energy** of the C-S bond and C-S **vibrational frequency**

Features demonstrated in this example

1. Using QChem interface to read-in QM energies
2. Using GULP interface to perform single-point calculations and read-in energy
3. Using utils.reaxff to construct GULP-compatible reaxff library files


rxff-parallel
----------------
Parallel optimization of ReaxFF forcefield for a thio-ketone monomer against 2 objectives -- **Dissociation energy** of the C-S bond and C-S **vibrational frequency**

Features demonstrated in this example

1. All features from rxff-serial
2. Using utils.reaxff methods for generating forcefields templates and variable range files
