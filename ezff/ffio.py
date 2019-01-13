"""This module provide methods to handle reading and writing forcefields"""
import numpy as np
import time

def read_variable_bounds(filename, verbose=False):
    """Read permissible lower and upper bounds for decision variables used in forcefields optimization

    :param filename: Name of text file listing bounds for each decision variable that must be optimized
    :type filename: str

    :param verbose: Print all variables read-in
    :type verbose: bool
    """
    variable_bounds = {}
    while True: # Force-read the parameter bounds file. This will loop until something is read-in. This is required if multiple ranks access the same file at the same time
        time.sleep(np.random.rand())
        with open(filename, 'r') as variable_bounds_file:
            for line in variable_bounds_file:
                items = line.strip().split()
                key, values = items[0], items[1:]
                if key[0] == '_':
                    variable_bounds[key] = list(map(int, values))
                else:
                    variable_bounds[key] = list(map(float, values))
        if not variable_bounds == {}:
            break

    if verbose:
        allkeys = ''
        for key in variable_bounds:
            allkeys += str(key) + ', '
        print('Keys: ' + allkeys[:-2] + ' read from ' + filename)

    return variable_bounds



def read_forcefield_template(template_filename):
    """Read-in the forcefield template. The template is constructed from a functional forcefield file by replacing all optimizable numerical values with variable names enclosed within dual angled brackets << and >>.

    :param template_filename: Name of the forcefield template file to be read-in
    :type template_filename: str
    """
    while True: # Force-read the template forcefield. This will loop until something is read-in. This is required if multiple ranks read the same file at the same time
        time.sleep(np.random.rand())
        with open(template_filename) as forcefield_template_file:
            template_string = forcefield_template_file.read()
        if len(template_string) > 0:
            break

    return template_string



def write_forcefield_file(filename, template_string, parameters, verbose=False):
    """Generate a new forcefield from the template by replacing variables with numerical values

    :param filename: Name of the forcefield file to be written
    :type filename: str

    :param template_string: Text of the forcefield template
    :type template_string: str

    :param parameters: Numerical value of all decision variables in the form of variable:value pairs
    :type parameters: dict

    :param verbose: Print one line confirming forcefield write-out
    :type verbose: bool
    """
    replaced_keys = ''
    for key, ranges in parameters.items():
        pattern = '<<' + key + '>>'
        while pattern in template_string:
            template_string = template_string.replace(pattern, '%12.6f' % ranges)
            if verbose:
                replaced_keys += key + ', '

    with open(filename, 'w') as new_forcefield:
        new_forcefield.write(template_string)

    if verbose:
        print('Forcefield, ' + filename + ', generated with new values for ' + replaced_keys[:-2])
