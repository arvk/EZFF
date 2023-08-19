# EZFF

[![Build Status](https://travis-ci.org/arvk/EZFF.svg?branch=master)](https://travis-ci.org/arvk/EZFF) [![Documentation Status](https://readthedocs.org/projects/ezff/badge/?version=latest)](https://ezff.readthedocs.io/en/latest/?badge=latest) [![Coverage Status](https://coveralls.io/repos/github/arvk/EZFF/badge.svg?branch=master)](https://coveralls.io/github/arvk/EZFF?branch=master) [![PyPI version](https://badge.fury.io/py/EZFF.svg)](https://badge.fury.io/py/EZFF)

Python-based library for easy force-field fitting

## Installation
Install from PyPI using
```
python3 -m pip install https://github.com/ppgaluzio/MOBOpt/archive/master.zip
python3 -m pip install EZFF
```

To get the examples and documentation, install the latest developmental version from GitHub via
```
git clone https://github.com/arvk/EZFF.git
cd EZFF
python setup.py install
```

## Documentation
Code documentation and examples can be found at [ezff.readthedocs.io](https://ezff.readthedocs.io/en/latest/)

## Requirements
Uses [xtal](https://github.com/USCCACS/xtal) for handling atomic structures and trajectories.
Multi-objective optimization is implemented through [Platypus](https://github.com/Project-Platypus/Platypus).

## Contributing
1. Please make sure to submit only passing builds
2. Adhere to PEP 8 if you can
3. Submit a pull request
