"""Interface to GULP, the General Utility Lattice Program"""
import os
import xtal
import numpy as np
import ezff
from ezff.ffio import generate_forcefield as gen_ff
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
            "phonon_dispersion_to": None,
            "phonon_G":None, # Phonons at gamma point
            "sshift": None
            }
        self.verbose = verbose
        if verbose:
            print('Created a new GULP job')

    @classmethod
    def from_dict(cls, root, data:dict):
        # only PBC
        job = cls(path=os.path.join(os.path.abspath(root), data['name']))
        job['structure'] = data['structure']
        job.options['pbc'] = True
        job.options["relax_atoms"] = True
        job.options["relax_cell"] = True

        if data.get['phonon_G']:
            job.options['phonon_G'] = True
        else:
            pass

        if data['phonon_dispersion']:
            job.options['phonon_dispersion'] = True
        else:
            pass

        job.options["phonon_dispersion_from"] = data.get("phonon_dispersion_from")
        job.options["phonon_dispersion_to"] = data.get("phonon_dispersion_to")
        
        return job

    def generate_forcefield(self, template_string, parameters, FFtype = None, outfile = None):
        self.options['fftype'] = FFtype.upper()
        forcefield = gen_ff(template_string, parameters, FFtype = FFtype, outfile = outfile, MD = 'GULP')
        return forcefield


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

        self._write_script_file()

        system_call_command = command + ' < ' + self.scriptfile + ' > ' + self.outfile + ' 2> ' + self.outfile + '.runerror'

        if timeout is not None:
            system_call_command = 'timeout ' + str(timeout) + ' ' + system_call_command

        if self.verbose:
            print('cd '+ self.path + ' ; ' + system_call_command)
        os.system('cd '+ self.path + ' ; ' + system_call_command)



    def _write_script_file(self):
        """
        Write-out a complete GULP script file, ``job.scriptfile``, based on job parameters
        """
        opts = self.options
        script = open(self.scriptfile,'w')
        header_line = ''
        if opts['relax_atoms']:
            header_line += 'optimise c6 '

            if opts['relax_cell']:
                header_line += 'conp '
            else:
                header_line += 'conv '

        if opts['phonon_G']:
            header_line += 'phonon nononanal '

        if header_line == '':
            header_line = 'gradient '

        if opts['phonon_dispersion'] is not None:
            header_line += 'phonon nofrequency '

        if opts['atomic_charges']:
            header_line += 'qiterative '

        header_line += 'comp property '
        script.write(header_line + '\n')

        if opts["sshift"]:
            script.write(f'sshift {opts["sshift"]}\n')

        script.write('\n')
        script.write('\n')

        # Write forcefield into script
        script.write(self.forcefield)

        script.write('\n')
        script.write('\n')

        if opts['pbc']:
            for snapshot in self.structure.snaplist:
                script.write('vectors\n')
                script.write(np.array_str(self.structure.box).replace('[','').replace(']','') + '\n')
                script.write('Fractional\n')
                for atom in snapshot.atomlist:
                    positions = atom.element.title() + ' core '
                    positions += np.array_str(atom.fract).replace('[','').replace(']','')
                    positions += ' 0.0   1.0   0.0   1 1 1 \n'
                    script.write(positions)
                script.write('\n\n\n')
        else:
            for snapshot in self.structure.snaplist:
                script.write('Cartesian\n')
                for atom in snapshot.atomlist:
                    positions = atom.element.title() + ' core '
                    positions += np.array_str(atom.cart).replace('[','').replace(']','')
                    positions += ' 0.0   1.0   0.0   1 1 1 \n'
                    script.write(positions)
                script.write('\n\n\n')
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
        """
        Read energy from completed GULP job

        :returns: Energy of the input structure(s) in eV as a np.ndarray
        """
        return _read_energy(self.outfile)

    def read_elastic_moduli(self):
        """
        Read elastic modulus matrix from a completed GULP job

        :returns: 6x6 Elastic modulus matrix in GPa for each input structure, as a list
        """
        return _read_elastic_moduli(self.outfile)

    def read_phonon_dispersion(self, units='cm-1'):
        """
        Read phonon dispersion from a complete GULP job

        :returns: 2D np.array containing the phonon dispersion in THz
        """
        return _read_phonon_dispersion(self.outfile+'.disp', units=units)

    def read_atomic_charges(self):
        """
        Read atomic charge information from a completed GULP job file

        :returns: xtal.AtTraj object with optimized charge information
        """
        return _read_atomic_charges(self.outfile)

    def read_structure(self):
        """
        Read converged structure (cell and atomic positions) from the MD job

        :returns: xtal.AtTraj object with (optimized) individual structures as separate snapshots
        """
        return _read_structure(self.outfile, relax_cell=self.options['relax_cell'], initial_box=self.structure.box)

    def read_phonon_G(self, n):
        return _read_phonon_G(self.outfile, n=n)

    def read_bulk_shear(self):
        return _read_bulk_shear(self.outfile)

    def read_young_xyz(self):
        return _read_young_xyz(self.outfile)

def _read_young_xyz(outfilename):
    """
    Read Young's modulus from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: list of Youngs Moduli values in GPa or zeroes
    """
    yx, yy, yz = [0]*3
    with open(outfilename) as f:
        for line in f:
            if 'Youngs Moduli (GPa)' in line:
                break
    try:
        ym_ = line.split('=')[1].split()
    except:
        ym_ = ['*']

    for i in ym_:
        if '*' in i:
            break
    else:
        yx, yy, yz = list(map(float, line.split('=')[1].split()))

    return yx, yy, yz

def _read_phonon_G(outfilename, n):
    """
    Read phonons frequences at Gamma point from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :param n: Number of frequencies the function should return
    :returns: list of first n frequencies
    """
    freq = []

    with open(outfilename) as fr:
        data_ = fr.readlines()

    try:
        i = data_.index('  Frequencies (cm-1) [NB: Negative implies an imaginary mode]:\n') + 2
    except:
        return [-1000]*n

    while i != 0:
        if '--------' in data_[i]:
            i = 0
            continue
        if data_[i]:
            try:
                for j in list(map(float, data_[i].split())):
                    freq.append(j)
            except:
                freq = [float(-1000)]*n
                break
            i += 1
        else:
            i = 0
    return freq[:n]

def _read_bulk_shear(otfilename):
    """
    Read Bulk and shear moduli from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: Values of Bulk and shear moduli packed in a tuple
    """
    _bulk_value, _shear_value = float(0), float(0)
    with open(otfilename, 'r') as fr:
        try:
            for line in fr:
                if 'Bulk  Modulus (GPa)' in line:
                    _bulk_value = float(line.split()[4])
                if 'Shear Modulus (GPa)' in line:
                    _shear_value = float(line.split()[4])
        except:
            pass
    return _bulk_value, _shear_value

def _read_elastic_moduli(outfilename):
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
    if not moduli_array:
        moduli_array = np.array([np.zeros((6, 6))])

    return moduli_array

def _read_energy(outfilename):
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
    outfile.close()

    if not energy_in_eV:
        energy_in_eV.append(float(0))

    return np.array(energy_in_eV)



def _read_phonon_dispersion(phonon_dispersion_file, units='cm-1'):
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



def _read_atomic_charges(outfilename):
    """
    Read atomic charge information from a completed GULP job file

    :param outfilename: Filename of the GULP output file
    :type outfilename: str
    :returns: xtal object with optimized charge information
    """
    structure = xtal.AtTraj()
    structure.box = np.zeros((3,3))
    outfile = open(outfilename, 'r')

    while True:
        oneline = outfile.readline()
        if not oneline:  # EOF check
            break
        if 'Output for configuration' in oneline:
            snapshot = structure.create_snapshot(xtal.Snapshot)
        if 'Final charges from ReaxFF' in oneline:
            snapshot.atomlist = []
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            while True:
                charges = outfile.readline().strip().split()
                if charges[0][0] == '-':
                    break
                else:
                    atom = snapshot.create_atom(xtal.Atom)
                    atom.charge = float(charges[-1])
    return structure



def _read_structure(outfilename, relax_cell=True, initial_box=None, D='3'):
    import re
    pattern = r'\s{2}Final\s[\w]+\s*\/*[\w]*\scoordinates[\w\s]*\s:'
    """
    Read converged structure (cell and atomic positions) from the MD job
    :param outfilename: Path of file containing stdout of the GULP job
    :type outfilename: str
    :param relax_cell: Flag to identify if simulation cell was relaxed during the MD job
    :type relax_cell: boolean
    :param initial_box: Initial simulation cell used for the MD job
    :type initial_box: 3X3 np.ndarray
    :returns: xtal.AtTraj object with (optimized) individual structures as separate snapshots
    """
    relaxed = xtal.AtTraj()
    relaxed.box = np.zeros((3,3))
    if (not relax_cell) and (initial_box is not None):
        relaxed.box = initial_box

    # Read number of atoms
    outfile = open(outfilename, 'r')
    for line in outfile:
        if 'Number of irreducible atoms/shells' in line.strip():
            snapshot = relaxed.create_snapshot(xtal.Snapshot)
            num_atoms = int(line.strip().split()[-1])
            for atomID in range(num_atoms):
                atom = snapshot.create_atom(xtal.Atom)
                atom.cart = np.array([0.0,0.0,0.0])
                atom.fract = np.array([0.0,0.0,0.0])
    outfile.close()

    snapID = 0
    convert_to_cart = False
    outfile = open(outfilename, 'r')

    while True:
        oneline = outfile.readline()

        if not oneline: # break for EOF
            break

        if re.search(pattern, oneline):
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            dummyline = outfile.readline()
            data_line = outfile.readline().split()[1:4]
            dummyline = outfile.readline()
            while True:
                data = outfile.readline().strip().split()
                if data[0].isdigit():
                    atomID = int(data[0])-1
                    for i in range(3):
                        axisID = i
                        if data_line[i] == '(Angs)':
                            try:
                                coord = float(data[i + 3])/50
                            except:
                                coord = 10000
                            relaxed.snaplist[snapID].atomlist[atomID].fract[axisID] = coord
                        else:
                            try:
                                coord = float(data[i + 3])
                            except:
                                coord = 10000
                            relaxed.snaplist[snapID].atomlist[atomID].fract[axisID] = coord 
                            convert_to_cart = True
                elif data[0][0] == '-':
                    break

            snapID += 1
    outfile.close()

    if convert_to_cart:
        if relax_cell:
            outfile = open(outfilename, 'r')
            snapID = 0
            while True:
                oneline = outfile.readline()

                if not oneline: # break for EOF
                    break

                if 'Final Cartesian lattice vectors (Angstroms)' in oneline:
                    dummyline = outfile.readline()
                    for i in range(3):
                        # Добавил для обхода ошибки при некорректной структуре
                        line = outfile.readline()
                        try:
                            relaxed.box[i][0:3] = list(map(float, line.strip().split()))
                        except:
                            relaxed.box[i][0:3] = [float(1000), float(1000), float(1000)]
                    if D == '3':
                        pass
                    if D == '2':
                        relaxed.box[2][0:3] = [0, 0, 50]
                    relaxed.make_dircar_matrices()
                    relaxed.snaplist[snapID].dirtocar()
                    snapID += 1
            outfile.close()
        else:
            relaxed.make_dircar_matrices()
            relaxed.dirtocar()
        
        relaxed.box_to_abc()

    return relaxed
