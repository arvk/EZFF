.. EZFF documentation master file, created by
   sphinx-quickstart on Wed Jan  9 10:52:51 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

EZFF -- Easy Multi-objective Forcefield Optimization
====================================================

EZFF is a Python-based library for quick and easy parameterization of forcefields and interatomic potentials for molecular dynamics simulations. EZFF provides interfaces to popular atomistic simulation software, GULP, LAMMPS, VASP, RXMD, and QChem and uses Platypus for solving multi-objective optimization problems. Use the links below to get started.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   basic
   algorithms
   examples
   code_documentation


Installing
==========
Install from PyPI using the command

.. code:: shell

   pip install EZFF


Alternatively, you can install the latest developmental version from GitHub via

.. code:: shell

   git clone https://github.com/arvk/EZFF.git
   cd EZFF
   python setup.py install


Contributing
============
1. Please make sure to submit only passing builds
2. Adhere to PEP8 where you can
3. Submit a pull request


License
=======
EZFF source code and documentation is released under the MIT License


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
