"""Interface to GULP, the General Utility Lattice Program"""
import os
import xtal
import numpy as np
from ezff.utils import convert_units as convert

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
        self.scriptfile ='in.gulp'
        self.outfile = 'out.gulp'
        self.command = 'gulp'
        self.forcefield = ''
        self.temporary_forcefield = False
        self.structure = None
        self.pbc = False
        self.options = {
            "relax_atoms": False,
            "relax_cell": False,
            "phonon_dispersion": None,
            "phonon_dispersion_from": None,
            "phonon_dispersion_to": None
            }
        self.verbose = verbose
        if verbose:
            print('Created a new GULP job')

    def run(self, command = None, parallel = False, processors = 1, timeout = None):
        """
        Execute GULP job with user-defined parameters

        :param command: path to GULP executable
        :type command: str

        :param parallel: Flag for parallel execution
        :type parallel: bool

        :param processors: Number of processors for parallel execution of each GULP job
        :type processors: int

        :param timeout: GULP job is automatically killed after ``timeout`` seconds
        :type timeout: int
        """
        if command is None:
            command = self.command

        if parallel:
            command = "mpirun -np " + str(processors) + " " + command

        system_call_command = command + ' < ' + self.scriptfile + ' > ' + self.outfile + ' 2> ' + self.outfile + '.runerror'

        if timeout is not None:
            system_call_command = 'timeout ' + str(timeout) + ' ' + system_call_command

        if self.verbose:
            print('cd '+ self.path + ' ; ' + system_call_command)
        os.system('cd '+ self.path + ' ; ' + system_call_command)


    def read_atomic_structure(self,structure_file):
        """
        Read-in atomic structure. Currently only VASP POSCAR/CONTCAR files are supported

        :param structure_file: Filename of the atomic structure file
        :type structure_file: str
        :returns: xtal trajectory with the structure in the first snapshot
        """
        structure = xtal.AtTraj(verbose=False)

        if ('POSCAR' in structure_file) or ('CONTCAR' in structure_file):
            structure.read_snapshot_vasp(structure_file)

        return structure


    def write_script_file(self, convert_reaxff=None):
        """
        Write-out a complete GULP script file, ``job.scriptfile``, based on job parameters

        :param convert_reaxff: Optional function that manipulates ``job.forcefield`` before the script file is written
        :type convert_reaxff: bool
        """
        opts = self.options
        script = open(self.path+'/'+self.scriptfile,'w')
        header_line = ''
        if opts['relax_atoms']:
            header_line += 'optimise '

            if opts['relax_cell']:
                header_line += 'conp '
            else:
                header_line += 'conv '

        if opts['phonon_dispersion'] is not None:
            header_line += 'phonon nofrequency '

        if header_line == '':
            header_line = 'single '

        header_line += 'comp property '
        script.write(header_line + '\n')

        script.write('\n')

        if self.pbc:
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


        if convert_reaxff is None:
            with open(self.forcefield,'r') as forcefield_file:
                forcefield = forcefield_file.read()
            for line in forcefield.split('\n'):
                script.write(' '.join(line.split()) + '\n')
        else:
            self.forcefield = convert_reaxff(self.forcefield)
            script.write('library ' + os.path.basename(self.forcefield))
        script.write('\n')

        if opts['phonon_dispersion_from'] is not None:
            if opts['phonon_dispersion_to'] is not None:
                script.write('dispersion 1 100 \n')
                script.write(opts['phonon_dispersion_from'] + ' to ' + opts['phonon_dispersion_to']+'\n')
                script.write('output phonon ' + self.outfile)
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
        if self.temporary_forcefield:
            if os.path.isfile(self.forcefield):
                os.remove(self.forcefield)



def read_elastic_moduli(outfilename):
    """
    Read elastic modulus matrix from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: 6x6 Elastic modulus matrix in GPa
    """
    moduli = np.zeros((6,6))
    outfile = open(outfilename,'r')
    for oneline in outfile:
        if 'Elastic Constant Matrix' in oneline:
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
            break
    outfile.close()
    return moduli


def read_lattice_constant(outfilename):
    """
    Read lattice constant values from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: Dictionary with ``abc`` - 3 lattice constants, ``ang`` - 3 supercell angles, ``err_abc`` - error in lattice constant, ``err_ang`` - error in supercell angles
    """
    abc, ang = np.zeros(3), np.zeros(3)
    err_abc, err_ang = np.zeros(3), np.zeros(3)
    outfile = open(outfilename, 'r')
    for oneline in outfile:
        if 'Comparison of initial and final' in oneline:
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
                    break
            break
    outfile.close()
    lattice = {'abc': abc, 'ang': ang, 'err_abc': err_abc, 'err_ang': err_ang}
    return lattice

def read_energy(outfilename):
    """
    Read single-point from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: Energy of the structure in eV
    """
    outfile = open(outfilename, 'r')
    for line in outfile:
        if 'Total lattice energy' in line:
            if line.strip().split()[-1] == 'eV':
                energy_in_eV = float(line.strip().split()[-2])
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



def error_structure_distortion(outfilename, relax_atoms=False, relax_cell=False):
    if not relax_atoms:  # If atoms are not relaxed (i.e. single point calculation, then return 0.0)
        return 0.0       # In the future, add a info/warning message in the log

    if relax_atoms:      # If atoms are leaxed, then create 2 atomic trajectories, one for each of the initial and relaxed structures
        initial = xtal.AtTraj()
        initial.abc = np.array([0.0,0.0,0.0])
        initial.ang = np.array([0.0,0.0,0.0])
        initial_snapshot = initial.create_snapshot(xtal.Snapshot)
        relaxed = xtal.AtTraj()
        relaxed.abc = np.array([0.0,0.0,0.0])
        relaxed.ang = np.array([0.0,0.0,0.0])
        relaxed_snapshot = relaxed.create_snapshot(xtal.Snapshot)

        # Read number of atoms
        outfile = open(outfilename, 'r')
        for line in outfile:
            if 'Number of irreducible atoms/shells' in line.strip():
                num_atoms = int(line.strip().split()[-1])
        outfile.close()

        for atomID in range(num_atoms):
            initial_snapshot.create_atom(xtal.Atom)
            initial_snapshot.atomlist[atomID].cart = np.array([0.0,0.0,0.0])
            initial_snapshot.atomlist[atomID].fract = np.array([0.0,0.0,0.0])
            relaxed_snapshot.create_atom(xtal.Atom)
            relaxed_snapshot.atomlist[atomID].cart = np.array([0.0,0.0,0.0])
            relaxed_snapshot.atomlist[atomID].fract = np.array([0.0,0.0,0.0])

        if relax_cell:     # In atoms are relaxed, and simulation cell is also relaxed
            convert_to_cart = True # We have to read in 2 box sizes, one for the initial cell and one for the relaxed cell
            outfile = open(outfilename, 'r')
            for oneline in outfile:
                if 'Comparison of initial and final' in oneline:
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    while True:
                        data = outfile.readline().strip().split()
                        if data[0].isdigit():
                            atomID = int(data[0])-1
                            if data[1] == 'x':
                                axisID = 0
                            elif data[1] == 'y':
                                axisID = 1
                            else:
                                axisID = 2

                            if data[5] == 'Cartesian':
                                initial_snapshot.atomlist[atomID].cart[axisID] = float(data[2])
                                relaxed_snapshot.atomlist[atomID].cart[axisID] = float(data[3])
                                convert_to_cart = False
                            elif data[5] == 'Fractional':
                                initial_snapshot.atomlist[atomID].fract[axisID] = float(data[2])
                                relaxed_snapshot.atomlist[atomID].fract[axisID] = float(data[3])
                        elif data[0][0] == '-':
                            break
                    break
            outfile.close()

            if convert_to_cart: # READ TWO CELL SIZES - ONE FOR THE INITIAL CELL, ONE FOR THE RELAXED CELL
                outfile = open(outfilename, 'r')
                for oneline in outfile:
                    if 'Comparison of initial and final' in oneline:
                        dummyline = outfile.readline()
                        dummyline = outfile.readline()
                        dummyline = outfile.readline()
                        dummyline = outfile.readline()
                        while True:
                            data = outfile.readline().strip().split()
                            if data[0] == 'a':
                                initial.abc[0], relaxed.abc[0] = float(data[1]), float(data[2])
                            elif data[0] == 'b':
                                initial.abc[1], relaxed.abc[1] = float(data[1]), float(data[2])
                            elif data[0] == 'c':
                                initial.abc[2], relaxed.abc[2] = float(data[1]), float(data[2])
                            elif data[0] == 'alpha':
                                initial.ang[0], relaxed.ang[0] = float(data[1]), float(data[2])
                            elif data[0] == 'beta':
                                initial.ang[1], relaxed.ang[1] = float(data[1]), float(data[2])
                            elif data[0] == 'gamma':
                                initial.ang[2], relaxed.ang[2] = float(data[1]), float(data[2])
                            elif data[0][0] == '-':
                                break
                        break
                outfile.close()
                initial.abc_to_box()
                relaxed.abc_to_box()
                initial.make_dircar_matrices()
                relaxed.make_dircar_matrices()
                initial.dirtocar()
                relaxed.dirtocar()
                relaxed.move(initial.snaplist[0].atomlist[0].cart  - relaxed.snaplist[0].atomlist[0].cart)
                error = 0.0
                for i in range(len(initial.snaplist[0].atomlist)):
                    dr = initial.snaplist[0].atomlist[i].cart - relaxed.snaplist[0].atomlist[i].cart
                    error += np.inner(dr, dr)
            else:
                relaxed.move(initial.snaplist[0].atomlist[0].cart  - relaxed.snaplist[0].atomlist[0].cart)
                error = 0.0
                for i in range(len(initial.snaplist[0].atomlist)):
                    dr = initial.snaplist[0].atomlist[i].cart - relaxed.snaplist[0].atomlist[i].cart
                    error += np.inner(dr, dr)
            return error

        else:     # IF THE CELL IS NOT RELAXED. JUST THE ATOMS ARE RELAXED
            convert_to_cart = True
            outfile = open(outfilename, 'r')
            for oneline in outfile:
                if 'Comparison of initial and final' in oneline:
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    dummyline = outfile.readline()
                    while True:
                        data = outfile.readline().strip().split()
                        if data[0].isdigit():
                            atomID = int(data[0])-1
                            if data[1] == 'x':
                                axisID = 0
                            elif data[1] == 'y':
                                axisID = 1
                            else:
                                axisID = 2

                            if data[5] == 'Cartesian':
                                initial_snapshot.atomlist[atomID].cart[axisID] = float(data[2])
                                relaxed_snapshot.atomlist[atomID].cart[axisID] = float(data[3])
                                convert_to_cart = False
                            elif data[5] == 'Fractional':
                                initial_snapshot.atomlist[atomID].fract[axisID] = float(data[2])
                                relaxed_snapshot.atomlist[atomID].fract[axisID] = float(data[3])
                        elif data[0][0] == '-':
                            break
                    break
            outfile.close()

            if convert_to_cart:
                outfile = open(outfilename, 'r')   ### READ A SINGLE BOX SIZE FOR THE INITIAL AND RELAXED STRUCTURES
                for oneline in outfile:
                    if 'Cartesian lattice vectors (Angstroms)' in oneline:
                        dummyline = outfile.readline()
                        for i in range(3):
                            initial.box[i][0:3] = list(map(float,outfile.readline().strip().split()))
                outfile.close()
                relaxed.box = initial.box
                initial.make_dircar_matrices()
                relaxed.make_dircar_matrices()
                initial.dirtocar()
                relaxed.dirtocar()
                relaxed.move(initial.snaplist[0].atomlist[0].cart  - relaxed.snaplist[0].atomlist[0].cart)
                error = 0.0
                for i in range(len(initial.snaplist[0].atomlist)):
                    dr = initial.snaplist[0].atomlist[i].cart - relaxed.snaplist[0].atomlist[i].cart
                    error += np.inner(dr, dr)
            else:
                relaxed.move(initial.snaplist[0].atomlist[0].cart  - relaxed.snaplist[0].atomlist[0].cart)
                error = 0.0
                for i in range(len(initial.snaplist[0].atomlist)):
                    dr = initial.snaplist[0].atomlist[i].cart - relaxed.snaplist[0].atomlist[i].cart
                    error += np.inner(dr, dr)
            return error
