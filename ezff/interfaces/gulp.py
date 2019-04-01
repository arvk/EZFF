"""Interface to GULP, the General Utility Lattice Program"""
import os
import xtal
import numpy as np
import ezff
from ezff.utils import convert_units as convert
import distutils
from distutils import spawn

class job:
    """
    Class representing a GULP calculation
    """

    def __init__(self, verbose=False, path='.'):
        """
        :param path: Path where the GULP job must be run from
        :type path: str

        :param verbose: Print details about the GULP job
        :type verbose: bool
        """
        if not os.path.isdir(path):
            if verbose:
                print('Path for current job is not valid . Creating a new directory...')
            os.makedirs(path)
        self.path = path

        self.scriptfile = os.path.join(os.path.abspath(path), 'in.gulp')
        self.outfile = os.path.join(os.path.abspath(path), 'out.gulp')
        self.structure = None
        self.forcefield = ''
        self.options = {
            "relax_atoms": False,
            "relax_cell": False,
            "pbc": False,
            "atomic_charges": False,
            "phonon_dispersion": None,
            "phonon_dispersion_from": None,
            "phonon_dispersion_to": None
            }
        self.verbose = verbose
        if verbose:
            print('Created a new GULP job')

    def run(self, command = None, timeout = None):
        """
        Execute GULP job with user-defined parameters

        :param command: path to GULP executable
        :type command: str

        :param timeout: GULP job is automatically killed after ``timeout`` seconds
        :type timeout: int
        """
        if command is None:
            # Attempt to locate a `gulp` executable
            gulpexec = distutils.spawn.find_executable('gulp')
            if gulpexec is not None:
                print('Located GULP executable at ' + gulpexec)
                command = gulpexec
            else:
                print('No GULP executable specified or located')

        self.write_script_file()

        system_call_command = command + ' < ' + self.scriptfile + ' > ' + self.outfile + ' 2> ' + self.outfile + '.runerror'

        if timeout is not None:
            system_call_command = 'timeout ' + str(timeout) + ' ' + system_call_command

        if self.verbose:
            print('cd '+ self.path + ' ; ' + system_call_command)
        os.system('cd '+ self.path + ' ; ' + system_call_command)


    def write_script_file(self):
        """
        Write-out a complete GULP script file, ``job.scriptfile``, based on job parameters
        """
        opts = self.options
        script = open(self.scriptfile,'w')
        header_line = ''
        if opts['relax_atoms']:
            header_line += 'optimise '

            if opts['relax_cell']:
                header_line += 'conp '
            else:
                header_line += 'conv '

        if opts['phonon_dispersion'] is not None:
            header_line += 'phonon nofrequency '

        if opts['atomic_charges']:
            header_line += 'qeq '

        if header_line == '':
            header_line = 'single '

        header_line += 'comp property '
        script.write(header_line + '\n')

        script.write('\n')
        script.write('\n')

        # Write forcefield into script
        script.write(self.forcefield)

        script.write('\n')
        script.write('\n')

        if opts['pbc']:
            script.write('vectors\n')
            script.write(np.array_str(self.structure.box).replace('[','').replace(']','') + '\n')
            script.write('Fractional\n')
            for atom in self.structure.snaplist[0].atomlist:
                positions = atom.element.title() + ' core '
                positions += np.array_str(atom.fract).replace('[','').replace(']','')
                positions += ' 0.0   1.0   0.0   1 1 1 \n'
                script.write(positions)
        else:
            script.write('Cartesian\n')
            for atom in self.structure.snaplist[0].atomlist:
                positions = atom.element.title() + ' core '
                positions += np.array_str(atom.cart).replace('[','').replace(']','')
                positions += ' 0.0   1.0   0.0   1 1 1 \n'
                script.write(positions)
        script.write('\n')


        if opts['phonon_dispersion_from'] is not None:
            if opts['phonon_dispersion_to'] is not None:
                script.write('dispersion 1 100 \n')
                script.write(opts['phonon_dispersion_from'] + ' to ' + opts['phonon_dispersion_to']+'\n')
                script.write('output phonon ' + os.path.basename(self.outfile + '.disp'))
                script.write('\n')

        script.write('\n')

        script.close()


    def cleanup(self):
        """
        Clean-up after the completion of a GULP job. Deletes input, output and forcefields files
        """
        files_to_be_removed = [self.outfile+'.disp', self.outfile+'.dens', self.outfile, self.scriptfile, self.outfile+'.runerror']
        for file in files_to_be_removed:
            if os.path.isfile(file):
                os.remove(file)
            elif os.path.isfile(self.path+'/'+file):
                os.remove(self.path+'/'+file)


    ## OOP methods for reading output from GULP
    def read_energy(self):
        return read_energy(self.outfile)

    def read_elastic_moduli(self):
        return read_elastic_moduli(self.outfile)

    def read_lattice_constant(self):
        return read_lattice_constant(self.outfile)

    def read_phonon_dispersion(self, units='cm-1'):
        return read_phonon_dispersion(self.outfile+'.disp', units=units)

    def error_structure_distortion(self):
        return ezff.error_structure_distortion(self.outfile, relax_atoms=self.options['relax_atoms'], relax_cell=self.options['relax_cell'])




def read_elastic_moduli(outfilename):
    """
    Read elastic modulus matrix from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: 6x6 Elastic modulus matrix in GPa
    """
    outfile = open(outfilename,'r')
    moduli_array = []
    while True:
        oneline = outfile.readline()
        if not oneline: # break at EOF
            break
        if 'Elastic Constant Matrix' in oneline:
            moduli = np.zeros((6,6))
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            for i in range(6):
                modline = outfile.readline().strip()
                e1, e2, e3, e4, e5, e6 = modline[3:13], modline[13:23], modline[23:33], modline[33:43], modline[43:53], modline[53:63]
                modarray = [e1,e2,e3,e4,e5,e6]
                float_modarray = []
                # Handle errors
                for element in modarray:
                    if element[0] == "*":
                        float_modarray.append(0.0)
                    else:
                        float_modarray.append(float(element))
                moduli[i,:] = float_modarray
            moduli_array.append(moduli)
    outfile.close()
    return moduli_array


def read_lattice_constant(outfilename):
    """
    Read lattice constant values from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: Dictionary with ``abc`` - 3 lattice constants, ``ang`` - 3 supercell angles, ``err_abc`` - error in lattice constant, ``err_ang`` - error in supercell angles
    """
    abc, ang = np.zeros(3), np.zeros(3)
    err_abc, err_ang = np.zeros(3), np.zeros(3)
    lattice_array = []
    outfile = open(outfilename, 'r')
    while True:
        oneline = outfile.readline()
        if not oneline:  # EOF check
            break
        if 'Comparison of initial and final' in oneline:
            lattice = {}
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            while True:
                data = outfile.readline().strip().split()
                if data[0] == 'a':
                    abc[0], err_abc[0] = data[2], data[-1]
                elif data[0] == 'b':
                    abc[1], err_abc[1] = data[2], data[-1]
                elif data[0] == 'c':
                    abc[2], err_abc[2] = data[2], data[-1]
                elif data[0] == 'alpha':
                    ang[0], err_ang[0] = data[2], data[-1]
                elif data[0] == 'beta':
                    ang[1], err_ang[1] = data[2], data[-1]
                elif data[0] == 'gamma':
                    ang[2], err_ang[2] = data[2], data[-1]
                elif data[0][0] == '-':
                    lattice = {'abc': abc, 'ang': ang, 'err_abc': err_abc, 'err_ang': err_ang}
                    lattice_array.append(lattice)
                    break
    outfile.close()

    return lattice_array



def read_energy(outfilename):
    """
    Read single-point from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: Energy of the structure in eV
    """
    energy_in_eV = []
    outfile = open(outfilename, 'r')
    for line in outfile:
        if 'Total lattice energy' in line:
            if line.strip().split()[-1] == 'eV':
                energy_in_eV.append(float(line.strip().split()[-2]))
    return energy_in_eV
    outfile.close()



def read_phonon_dispersion(phonon_dispersion_file, units='cm-1'):
    """
    Read phonon dispersion from a complete GULP job

    :param phonon_dispersion_file: Path of file containing phonon dispersion from the GULP job
    :type phonon_dispersion_file: str
    :returns: 2D np.array containing the phonon dispersion in THz
    """
    pbs = []
    freq_conversion = {'cm-1': 0.0299792453684314, 'THz': 1, 'eV': 241.79893, 'meV': 0.24180}
    dispersion = open(phonon_dispersion_file, 'r')
    for line in dispersion:
        if not line.strip().startswith('#'):
            str_index, str_freq = line[:4], line[4:]
            if not str_freq[0] == '*':
                pbs.append(float(str_freq))
            else:
                pbs.append(0.0)
    dispersion.close()
    num_bands = int(len(pbs)/100)
    pbs = np.array(pbs).reshape((100,num_bands)).T
    pbs *= convert.frequency[units]['THz']
    return pbs



def read_atomic_charges(outfilename):
    """
    Read atomic charge information from a completed GULP job file

    :param outfilename: Filename of the GULP output file
    :type outfilename: str
    :returns: xtal object with optimized charge information
    """

    structure = xtal.AtTraj()
    snapshot = structure.create_snapshot(xtal.Snapshot)
    outfile = open(outfilename, 'r')

    natoms = None

    for line in outfile:

        if 'Total number atoms/shells' in line:
            natoms = int(line.strip().split()[-1])

        if 'Final charges from QEq' in line:
            snapshot.atomlist = []
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            # Atomic Charge information starts here
            counter = 0
            while True:
                charges = outfile.readline()
                charges = charges.strip().split()
                atom = snapshot.create_atom(xtal.Atom)
                if float(charges[1]) == 1:
                    atom.element = 'H'
                if float(charges[1]) == 6:
                    atom.element = 'C'
                if float(charges[1]) == 7:
                    atom.element = 'N'
                if float(charges[1]) == 8:
                    atom.element = 'O'

                atom.charge = float(charges[2])

                counter += 1
                if counter == natoms:
                    break
    return structure
