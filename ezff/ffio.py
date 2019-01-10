"""This module provide methods to handle reading and writing forcefields"""

def read_parameter_bounds(filename, verbose=False):
    """Read permissible lower and upper bounds for decision variables used in forcefields optimization

    :param filename: Name of text file listing bounds for each decision variable that must be optimized
    :type filename: str

    :param verbose: Print all variables read-in
    :type verbose: bool
    """
    parameter_bounds = {}
    with open(filename, 'r') as parameter_bounds_file:
        for line in parameter_bounds_file:
            items = line.strip().split()
            key, values = items[0], items[1:]
            if key[0] == '_':
                parameter_bounds[key] = list(map(int, values))
            else:
                parameter_bounds[key] = list(map(float, values))

    if verbose:
        allkeys = ''
        for key in parameter_bounds:
            allkeys += str(key) + ', '
        print('Keys: ' + allkeys[:-2] + ' read from ' + filename)

    return parameter_bounds



def read_parameter_template(template_filename):
    """Read-in the forcefield template. The template is constructed from a functional forcefield file by replacing all optimizable numerical values with variable names enclosed within dual angled brackets << and >>.

    :param template_filename: Name of the forcefield template file to be read-in
    :type template_filename: str
    """
    with open(template_filename) as parameter_template_file:
        template_string = parameter_template_file.read()
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