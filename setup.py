from setuptools import setup, find_packages

setup(name='EZFF',
      version='1.0.5', # Update __init__.py if the version changes!
      description='Multiobjective forcefield optimization for Molecular Dynamics',
      author='Aravind Krishnamoorthy',
      author_email='arvk@users.noreply.github.com',
      license='MIT License',
      url='https://github.com/arvk/EZFF',
      packages=find_packages(exclude=["docs", "examples"]),
      install_requires=[
          'mpi4py >= 3.0.0',
          'xtal >= 0.9.1',
          'schwimmbad >= 0.3.2',
          'nevergrad >= 0.11.0',
          'pymoo == 0.6.0.dev0',
          'platypus-opt >= 1.0.3',
          'deap >= 1.3.0',
          'periodictable >= 1.6.1',
          'cclib >= 1.8',
      ],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Chemistry',
          'Topic :: Scientific/Engineering :: Physics',
          'Natural Language :: English',
          'Operating System :: POSIX :: Linux',
          'Operating System :: Unix',
      ],
     )
