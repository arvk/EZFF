Basic Usage
===========
You will need five pieces of information to get started with focefield optimization using EZFF.

1. Ground truths - Physical properties to parameterize the forcefield against
2. Serial MD executable (either GULP or LAMMPS or RXMD)
3. Forcefield template
4. Maximum and minimum values of decision variables
5. A master python script (`run.py` in examples) that defines different errors, handles GULP jobs and optimization parameters
6. (**Optional**) A working installation of MPI and mpi4py for parallel optimization


:Ground truths:
   Ground truth values (for lattice constant, elastic constants, energies, phonon dispersion curves etc) can either be provided by hand in the master script, or can be calculated using methods provided in the different ezff.interfaces modules

:Serial MD executable:
   A working serial MD engine. Options available are:

   1. Serial GULP executable built from source code available from http://gulp.curtin.edu.au/gulp/
   2. Serial LAMMPS executable built from https://www.lammps.org/#gsc.tab=0
   3. Serial RXMD executable built from https://magics.usc.edu/rxmd/
   Parallel optimization jobs simply launch multiple copies of the serial executable.

:Forcefield template:
   A forcefield template file is used to designate which variables must be considered for optimization. The forcefield template is constructed from a functioning GULP-readable forcefield by replacing the parameters that need to be optimized with variable names enclosed within dual angle-brackets. For example, the following Lennard-Jones forcefield for solid Neon (from examples lj-serial and lj-parallel) ::

     lennard epsilon 12 6  # Tell GULP that the next line will contain LJ parameters
     Ne Ne 1.0 1.5  # Format: atom1 atom2 epsilon sigma

   can be converted to a template by replacing epsilon and sigma by optimizable decision variables::

     lennard epsilon 12 6  # Tell GULP that the next line will contain LJ parameters
     Ne Ne <<eps>> <<sgma>>  # Format: atom1 atom2 epsilon sigma

   This fill will instruct EZFF to optimize the **eps** and **sgma** variables. Decision variables are assumed to be real-valued by default. Any decision variable beginning with an underscore (_) is assumed to be integer-valued.

   The template forcefield must be paired with an appropriate file specifying the permissible ranges of these decision variables.

:Decision variable ranges:
   This file lists the permissible ranges of decision variables (i.e. minimum and maximum value that the variable can take) during forcefield optimization. This text file is written in the following format::

     Decision_variable_1(without the angle brackets)    Minimum_value  Maximum_value
     Decision_variable_2(without the angle brackets)    Minimum_value  Maximum_value
     Decision_variable_3(without the angle brackets)    Minimum_value  Maximum_value
     .
     .
     Decision_variable_n(without the angle brackets)    Minimum_value  Maximum_value

   A valid variable range file for the Lennard-Jones template file above is given below::

     eps 0.5 2.5
     sgma 0.1 0.4

   .. warning::
      Please ensure that the template and variable_ranges file are compatible. Specifically,

      1. Every variable defined in the forcefield template must have one (and only one) corresonding entry in the variable ranges file
      2. The variable ranges file should not refer to variables not present in the template file


:Python script:
   This python script should include, at the very least, your custom function to calculate the error (i.e. deviation of the forcefield from ground-truths), an ezff.FFParam object, a call to ezff.set_algorithm and a call to ezff.parameterize.

   The custom error function should be written to accept one input -- a dictionary of decision_variable-value pairs (e.g. {'eps': 1.273, 'sgma': 0.12} for example above) and should return a list of all computed objectives. The length of this returned list should equal the number of errors.
