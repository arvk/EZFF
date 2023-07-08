# EZFF

**Note** this code is modified version of [EZFF](https://github.com/arvk/EZFF)

Python-based library for easy force-field fitting

## Installation

To get the examples and documentation, install the latest developmental version from GitHub via
```
git clone https://github.com/akvatol/EZFF2
cd EZFF
python setup.py install
```

## Documentation
Code documentation and examples can be found at [ezff.readthedocs.io](https://ezff.readthedocs.io/en/latest/)

## Requirements
Uses [xtal](https://github.com/USCCACS/xtal) for handling atomic structures and trajectories.
Multi-objective optimization is implemented through [Platypus](https://github.com/Project-Platypus/Platypus).

## Modifications
Added the ability to calculate the phonon frequencies at the Gamma point, as well as to read the Young's modulus (see example ws2-serial).
