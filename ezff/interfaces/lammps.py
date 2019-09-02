"""Interface to LAMMPS, the Large-scale Atomic/Molecular Massively Parallel Simulator"""
import os
import xtal
import numpy as np
import ezff
from ezff.ffio import generate_forcefield as gen_ff
from ezff.utils import convert_units as convert
from ezff.utils import atomic_properties
import distutils
from distutils import spawn


class job:
    """
    Class representing a LAMMPS calculation
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

        self.scriptfile = os.path.join(os.path.abspath(path), 'in.lmp')
        self.outfile = os.path.join(os.path.abspath(path), 'out.lmp')
        self.structfile = os.path.join(os.path.abspath(path), 'input.structure')
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
            print('Created a new LAMMPS job')


    def generate_forcefield(self, template_string, parameters, FFtype = None, outfile = None):
        self.options['fftype'] = FFtype.upper()
        forcefield = gen_ff(template_string, parameters, FFtype = FFtype, outfile = outfile, MD = 'LAMMPS')
        return forcefield


    def run(self, command = None, timeout = None):
        """
        Execute LAMMPS job with user-defined parameters

        :param command: path to LAMMPS executable
        :type command: str

        :param timeout: LAMMPS job is automatically killed after ``timeout`` seconds
        :type timeout: int
        """
        if command is None:
            # Attempt to locate a `gulp` executable
            lmpexec = distutils.spawn.find_executable('lmp_serial')
            if lmpexec is not None:
                print('Located LAMMPS executable at ' + lmpexec)
                command = lmpexec
            else:
                print('No LAMMPS executable specified or located')

        ## Purge existing result files
        os.system('cd '+ self.path + ' ; rm ' + self.outfile)

        outfile_single_snapshot = self.outfile + '_temp'

        for snapID, snapshot in enumerate(self.structure.snaplist):
            self._write_structure_file(snapID)
            self._write_script_file()

            system_call_command = command + ' -in ' + self.scriptfile + ' > ' + outfile_single_snapshot + ' 2> ' + self.outfile + '.runerror'

            if timeout is not None:
                system_call_command = 'timeout ' + str(timeout) + ' ' + system_call_command

            if self.verbose:
                print('cd '+ self.path + ' ; ' + system_call_command)
            os.system('cd '+ self.path + ' ; ' + system_call_command)

            # Append output to the job outfile
            os.system('cat ' + outfile_single_snapshot + ' >> ' + self.outfile)


    def _write_structure_file(self, snap_ID):
        """
        Write-out a complete structure file for the LAMMPS ``read_data`` command, based on job parameters
        """
        opts = self.options
        self.structure.box_to_abc()
        snapshot = self.structure.snaplist[snap_ID]

        structfile = open(self.structfile,'w')

        ## Define mapping between atomic elements and numeric IDs --> Only numeric IDs are allowed in structure files. Remapping is done in the script file
        map_element_id = {}
        atom_types = set([atom.element for atom in snapshot.atomlist])
        for ID, atom_type in enumerate(atom_types):
            map_element_id.update({atom_type: ID+1})

        self.options['atom_sequence'] = list(atom_types)

        # Write header for the data file
        structfile.write('## LAMMPS structure file from EZFF \n\n')
        structfile.write(str(len(snapshot.atomlist)) + ' atoms \n\n')
        structfile.write(str(len(atom_types)) + ' atom types \n\n')


        # Coordinate transformation (taken from https://lammps.sandia.gov/doc/Howto_triclinic.html)
        self.structure.box_to_abc()
        abc = self.structure.abc
        ang = self.structure.ang
        lx = abc[0]
        xy = abc[1] * np.cos(ang[2])
        xz = abc[2] * np.cos(ang[1])
        ly = np.sqrt((abc[1]*abc[1]) - (xy*xy))
        yz = ((abc[1]*(abc[2]*np.cos(ang[0]))) - (xy*xz))/ly
        lz = np.sqrt((abc[2]*abc[2]) - (xz*xz) - (yz*yz))


        # Correct for very large skew of the box
        xy = np.minimum(np.maximum(xy, 0.0-lx/2.0), lx/2.0)
        xz = np.minimum(np.maximum(xz, 0.0-lx/2.0), lx/2.0)
        yz = np.minimum(np.maximum(yz, 0.0-ly/2.0), ly/2.0)

        if ((abs(xy) < 1e-4) and (abs(yz) < 1e-4) and (abs(xz) < 1e-4)):
            structfile.write('0.0 ' + str(lx) + ' xlo xhi \n')
            structfile.write('0.0 ' + str(ly) + ' ylo yhi \n')
            structfile.write('0.0 ' + str(lz) + ' zlo zhi \n')
        else:
            structfile.write('0.0 ' + str(lx) + ' xlo xhi \n')
            structfile.write('0.0 ' + str(ly) + ' ylo yhi \n')
            structfile.write('0.0 ' + str(lz) + ' zlo zhi \n')
            structfile.write(str(xy) + ' ' + str(xz) + ' ' + str(yz) + ' xy xz yz \n')
        structfile.write('\n\n')

        structfile.write('Masses\n\n')
        for ID, atom_type in enumerate(atom_types):
            structfile.write('%d  %f \n' % (ID+1, atomic_properties.atomic_mass[atom_type]))
        structfile.write('\n\n')

        structfile.write('Atoms\n\n')
        for atomID, atom in enumerate(snapshot.atomlist):
            structfile.write('%d %d %f %f %f %f \n' % (atomID+1, map_element_id[atom.element], atom.charge, atom.cart[0], atom.cart[1], atom.cart[2]))


    def _write_forcefield_file(self):
        fffile = open(os.path.join(os.path.abspath(self.path), 'ff.lmp'),'w')
        fffile.write(self.forcefield)
        fffile.close()


    def _write_script_file(self):
        """
        Write-out a complete LAMMPS script file, ``job.scriptfile``, based on job parameters
        """
        opts = self.options
        script = open(self.scriptfile,'w')

        # Convert FFtype to pair_style
        fftype_pairstyle = {'SW': 'sw', 'STILLINGER-WEBER': 'sw', 'STILLINGER WEBER': 'sw',
                            'REAX': 'reax/c NULL', 'REAXFF': 'reax/c NULL', 'REAX/C': 'reax/c NULL',
                            'VASHISHTA': 'vashishta'}

        script.write('units metal \n')  # All simulations will be performed in LAMMPS metal units
        script.write('dimension 3 \n')  # All simulations will be 3D
        script.write('atom_style charge \n')

        if opts['pbc']:
            script.write('boundary p p p \n')
        else:
            script.write('boundary fm fm fm \n')

        script.write('read_data input.structure \n')
        script.write('pair_style ' + fftype_pairstyle[opts['fftype']] + '\n')
        script.write('pair_coeff * * ff.lmp ' + ' '.join(opts['atom_sequence']).title() + '\n')

        ## Define a universal thermo style and dump information every step
        script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz \n')
        script.write('thermo 1000 \n')

        script.write('variable ezff_T equal temp \n')
        script.write('variable ezff_V equal vol \n')
        script.write('variable ezff_E equal etotal \n')

        if opts['relax_atoms']:
            if opts['relax_cell']:
                script.write('fix FixBoxRelax all box/relax aniso 0.0 \n')
            script.write('minimize 0.0 1.0e-8 1000 100000 \n')
        else:
            script.write('run 0 \n')

        script.write('\n')
        script.write('\n')

        # Write out the summary
        script.write('write_dump all custom final.dump id type mass q x y z vx vy vz fx fy fz \n')

        script.write('print "-----SUMMARY-----" \n')
        script.write('print "EZFF_TEMP ${ezff_T}" \n')
        script.write('print "EZFF_VOL ${ezff_V}" \n')
        script.write('print "EZFF_ENERGY ${ezff_E}" \n')

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



def _read_structure(outfilename, relax_cell=True, initial_box=None):
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
    convert_to_cart = True
    outfile = open(outfilename, 'r')

    while True:
        oneline = outfile.readline()

        if not oneline: # break for EOF
            break

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
                        relaxed.snaplist[snapID].atomlist[atomID].cart[axisID] = float(data[3])
                        convert_to_cart = False
                    elif data[5] == 'Fractional':
                        relaxed.snaplist[snapID].atomlist[atomID].fract[axisID] = float(data[3])
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
                        relaxed.box[i][0:3] = list(map(float,outfile.readline().strip().split()))
                    relaxed.make_dircar_matrices()
                    relaxed.snaplist[snapID].dirtocar()
                    snapID += 1
            outfile.close()
        else:
            relaxed.make_dircar_matrices()
            relaxed.dirtocar()

    return relaxed
