from setuptools import setup, find_packages

setup(name='EZFF',
      version='0.9.3', # Update __init__.py if the version changes!
      description='Multiobjective forcefield optimization for Molecular Dynamics',
      author='Aravind Krishnamoorthy',
      author_email='arvk@users.noreply.github.com',
      license='MIT License',
      url='https://github.com/arvk/EZFF',
      packages=find_packages(exclude=["docs", "examples"]),
      install_requires=[
          'platypus-opt >= 1.0.3',
          'mpi4py >= 3.0.0',
          'xtal >= 0.9.0',
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
