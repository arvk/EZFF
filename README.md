# EZFF2

**Note** this code is modified version of [EZFF](https://github.com/arvk/EZFF)

Python-based library for easy force-field fitting

## Installation

=======
Install from PyPI using
```
python3 -m pip install https://github.com/ppgaluzio/MOBOpt/archive/master.zip
python3 -m pip install EZFF
```


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
Added the capability to calculate the phonon frequencies at the Gamma point and to read the Young's modulus (see to the example "ws2-serial").
