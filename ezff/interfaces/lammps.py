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
    def __init__(self, verbose=False, path='.', units='metal'):
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
        self.dumpfile = os.path.join(os.path.abspath(path), 'out.dump')
        self.structfile = os.path.join(os.path.abspath(path), 'input.structure')
        self.structure = None
        self.forcefieldfile = os.path.join(os.path.abspath(path), 'generated_forcefield')
        self.forcefield = ''
        self.units = units
        self.options = {
            "relax_atoms": False,
            "relax_cell": False,
            "pbc": False,
            "atomic_charges": False,
            "phonon_dispersion": None,
            "phonon_dispersion_from": None,
            "phonon_dispersion_to": None
            }
        self.pbc = True
        self.verbose = verbose
        if verbose:
            print('Created a new LAMMPS job')


    def generate_forcefield(self, template_string, parameters, FFtype = None, outfile = None):
        self.options['fftype'] = FFtype.upper()
        forcefield = gen_ff(template_string, parameters, FFtype = FFtype, outfile = outfile, MD = 'LAMMPS')
        return forcefield


    def run_static(self, command = None, timeout = None):
        """
        Execute LAMMPS job with user-defined parameters

        :param command: path to LAMMPS executable
        :type command: str

        :param timeout: LAMMPS job is automatically killed after ``timeout`` seconds
        :type timeout: int
        """
        self._pre_job_cleanup()
        for snapID, snapshot in enumerate(self.structure.snaplist):
            self._write_structure_file(snapID)
            self._write_script_file_static()
            self._run_loop(command = command, timeout = timeout)
        self._post_job_cleanup()



    def run_relax(self, command = None, timeout = None):
        """
        Execute LAMMPS job with user-defined parameters

        :param command: path to LAMMPS executable
        :type command: str

        :param timeout: LAMMPS job is automatically killed after ``timeout`` seconds
        :type timeout: int
        """
        self._pre_job_cleanup()
        for snapID, snapshot in enumerate(self.structure.snaplist):
            self._write_structure_file(snapID)
            self._write_script_file_relax()
            self._run_loop(command = command, timeout = timeout)
        self._post_job_cleanup()



    def run_elastic(self, command = None, timeout = None):
        """
        Execute LAMMPS job with user-defined parameters

        :param command: path to LAMMPS executable
        :type command: str

        :param timeout: LAMMPS job is automatically killed after ``timeout`` seconds
        :type timeout: int
        """
        self._pre_job_cleanup()
        for snapID, snapshot in enumerate(self.structure.snaplist):
            self._write_structure_file(snapID)
            self._write_script_file_elastic()
            self._run_loop(command = command, timeout = timeout)
        self._post_job_cleanup()



    def _pre_job_cleanup(self):
        os.system('cd '+ self.path + ' ; rm -f ' + self.outfile)


    def _run_loop(self, command = None, timeout = None):
        outfile_single_snapshot = self.outfile + '_temp'

        system_call_command = command + ' -in ' + self.scriptfile + ' > ' + outfile_single_snapshot + ' 2> ' + self.outfile + '.runerror'

        if timeout is not None:
            system_call_command = 'timeout ' + str(timeout) + ' ' + system_call_command

        if self.verbose:
            print('cd '+ self.path + ' ; ' + system_call_command)
        os.system('cd '+ self.path + ' ; ' + system_call_command)

        # Append output to the job outfile and structure to job structurefile
        os.system('cat ' + outfile_single_snapshot + ' >> ' + self.outfile)
        os.system('cat ' + os.path.join(os.path.abspath(self.path), 'tempdumpfile') + ' >> ' + self.dumpfile)
        os.system('rm -f ' + os.path.join(os.path.abspath(self.path), 'tempdumpfile'))
        os.system('rm -f ' + outfile_single_snapshot)


    def _post_job_cleanup(self):
        files_to_be_removed = [self.outfile+'.disp', self.outfile+'.dens', self.outfile+'.runerror']
        for file_to_be_removed in files_to_be_removed:
            abs_file_to_be_removed = os.path.join(os.path.abspath(self.path), file_to_be_removed)
            os.system('cd '+ self.path + ' ; rm -f ' + abs_file_to_be_removed)




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

            # Append output to the job outfile and structure to job structurefile
            os.system('cat ' + outfile_single_snapshot + ' >> ' + self.outfile)
            os.system('cat ' + os.path.join(os.path.abspath(self.path), 'tempdumpfile') + ' >> ' + self.dumpfile)
            os.system('rm -f tempdumpfile')


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

        xy = np.round(xy, decimals=4)
        xz = np.round(xz, decimals=4)
        yz = np.round(yz, decimals=4)

        # if ((abs(xy) < 1e-4) and (abs(yz) < 1e-4) and (abs(xz) < 1e-4)):
        #     structfile.write('0.0 ' + str(lx) + ' xlo xhi \n')
        #     structfile.write('0.0 ' + str(ly) + ' ylo yhi \n')
        #     structfile.write('0.0 ' + str(lz) + ' zlo zhi \n')
        # else:
        #     structfile.write('0.0 ' + str(lx) + ' xlo xhi \n')
        #     structfile.write('0.0 ' + str(ly) + ' ylo yhi \n')
        #     structfile.write('0.0 ' + str(lz) + ' zlo zhi \n')
        #     structfile.write(str(xy) + ' ' + str(xz) + ' ' + str(yz) + ' xy xz yz \n')


        structfile.write('0.0 ' + str(lx) + ' xlo xhi \n')
        structfile.write('0.0 ' + str(ly) + ' ylo yhi \n')
        structfile.write('0.0 ' + str(lz) + ' zlo zhi \n')
        structfile.write(str(xy) + ' ' + str(xz) + ' ' + str(yz) + ' xy xz yz \n')

        structfile.write('\n\n')

        structfile.write('Masses\n\n')
        for ID, atom_type in enumerate(atom_types):
            structfile.write('%d  %f \n' % (ID+1, atomic_properties.atomic_mass[atom_type.upper()]))
        structfile.write('\n\n')

        structfile.write('Atoms\n\n')
        for atomID, atom in enumerate(snapshot.atomlist):
            structfile.write('%d %d %f %f %f %f \n' % (atomID+1, map_element_id[atom.element], atom.charge, atom.cart[0], atom.cart[1], atom.cart[2]))


    def _write_forcefield_file(self):
        fffile = open(os.path.join(os.path.abspath(self.path), 'ff.lmp'),'w')
        fffile.write(self.forcefield)
        fffile.close()


        script.write('read_data input.structure \n')
        script.write('pair_style ' + fftype_pairstyle[opts['fftype']] + '\n')
        script.write('pair_coeff * * ff.lmp ' + ' '.join(opts['atom_sequence']).title() + '\n')

        ## Define a universal thermo style and dump information every step
        script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
        script.write('thermo 1000 \n')

        script.write('variable ezff_T equal temp \n')
        script.write('variable ezff_V equal vol \n')
        script.write('variable ezff_E equal etotal \n')

        script.write('run 0 \n')

        script.write('\n')
        script.write('\n')



    def _include_forcefield(self, fftype):
        # Convert FFtype to pair_style
        fftype_pairstyle = {'SW': 'sw', 'STILLINGER-WEBER': 'sw', 'STILLINGER WEBER': 'sw',
                            'REAX': 'reax/c NULL', 'REAXFF': 'reax/c NULL', 'REAX/C': 'reax/c NULL',
                            'VASHISHTA': 'vashishta'}
        return_string = ''
        if self.options['fftype'].upper() == 'LJ':
            return_string = 'include generated_forcefield \n'
            f = open(self.forcefieldfile, 'w')
            f.write(self.forcefield)
            f.close()
        elif self.options['fftype'].upper() == 'VASHISHTA':
            return_string = 'pair_style vashishta \n'
            return_string +=  'pair_coeff * * ' + self.forcefieldfile  + ' ' + ' '.join(self.options['atom_sequence']).title() + '\n'
            f = open(self.forcefieldfile, 'w')
            f.write(self.forcefield)
            f.close()
        elif 'REAX' in self.options['fftype'].upper():
            return_string = 'pair_style ' + fftype_pairstyle[self.options['fftype']] + '\n'
            return_string += 'pair_coeff * * ' + self.forcefieldfile + ' ' + ' '.join(self.options['atom_sequence']).title() + '\n'
            return_string += 'fix 1 all qeq/reaxff 1 0.0 10.0 1.0e-6 reaxff \n' # Set charge optimization
            f = open(self.forcefieldfile, 'w')
            f.write(self.forcefield)
            f.close()
        else:
            return_string = 'pair_style ' + fftype_pairstyle[self.options['fftype']] + '\n'
            return_string += 'pair_coeff * * ' + self.forcefieldfile + ' ' + ' '.join(self.options['atom_sequence']).title() + '\n'
            f = open(self.forcefieldfile, 'w')
            f.write(self.forcefield)
            f.close()
        return return_string




    def _write_script_file_static(self):
        opts = self.options
        script = open(self.scriptfile,'w')

        script.write('units ' +  self.units + ' \n')  # All simulations will be performed in LAMMPS metal units
        script.write('dimension 3 \n')  # All simulations will be 3D
        script.write('atom_style charge \n')

        if opts['pbc']:
            script.write('boundary p p p \n')
        else:
            script.write('boundary fm fm fm \n')

        script.write('read_data input.structure \n')

        ff_include_string = self._include_forcefield(self.options['fftype'])
        script.write(ff_include_string)

        ## Define a universal thermo style and dump information every step
        script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
        script.write('thermo 1000 \n')

        script.write('variable ezff_T equal temp \n')
        script.write('variable ezff_V equal vol \n')
        script.write('variable ezff_E equal etotal \n')

        script.write('run 0 \n')

        script.write('\n')
        script.write('\n')

        # Write out the summary
        script.write('write_dump all custom tempdumpfile id type element mass q x y z vx vy vz fx fy fz modify sort id element ' + ' '.join(opts['atom_sequence']).title() + ' \n')

        script.write('print "-----SUMMARY-----" \n')
        script.write('print "EZFF_TEMP ${ezff_T}" \n')
        script.write('print "EZFF_VOL ${ezff_V}" \n')
        script.write('print "EZFF_ENERGY ${ezff_E}" \n')

        script.close()





    def _write_script_file_relax(self):
        opts = self.options
        script = open(self.scriptfile,'w')

        script.write('units metal \n')  # All simulations will be performed in LAMMPS metal units
        script.write('dimension 3 \n')  # All simulations will be 3D
        script.write('atom_style charge \n')

        if opts['pbc']:
            script.write('boundary p p p \n')
        else:
            script.write('boundary fm fm fm \n')

        script.write('read_data input.structure \n')

        ff_include_string = self._include_forcefield(self.options['fftype'])
        script.write(ff_include_string)

        ## Define a universal thermo style and dump information every step
        script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
        script.write('thermo 1000 \n')

        script.write('variable ezff_T equal temp \n')
        script.write('variable ezff_V equal vol \n')
        script.write('variable ezff_E equal etotal \n')

        if opts['relax_cell']:
            script.write('fix FixBoxRelax all box/relax aniso 0.0 \n')
        script.write('minimize 0.0 1.0e-8 1000 100000 \n')

        script.write('run 0 \n')

        script.write('\n')
        script.write('\n')

        # Write out the summary
        script.write('write_dump all custom tempdumpfile id type element mass q x y z vx vy vz fx fy fz modify sort id element ' + ' '.join(opts['atom_sequence']).title() + ' \n')

        script.write('print "-----SUMMARY-----" \n')
        script.write('print "EZFF_TEMP ${ezff_T}" \n')
        script.write('print "EZFF_VOL ${ezff_V}" \n')
        script.write('print "EZFF_ENERGY ${ezff_E}" \n')

        script.close()





    def _write_script_file_elastic(self):
        opts = self.options
        script = open(self.scriptfile,'w')

        script.write('units metal \n')  # All simulations will be performed in LAMMPS metal units
        script.write('dimension 3 \n')  # All simulations will be 3D
        script.write('atom_style charge \n')

        if opts['pbc']:
            script.write('boundary p p p \n')
        else:
            script.write('boundary fm fm fm \n')

        script.write('read_data input.structure \n')

        ff_include_string = self._include_forcefield(self.options['fftype'])
        script.write(ff_include_string)

        ## Define a universal thermo style and dump information every step
        script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
        script.write('thermo 1000 \n')

        script.write('variable ezff_T equal temp \n')
        script.write('variable ezff_V equal vol \n')
        script.write('variable ezff_E equal etotal \n')

        if opts['relax_cell']:
            script.write('fix FixBoxRelax all box/relax aniso 0.0 \n')
        script.write('minimize 0.0 1.0e-8 1000 100000 \n')

        script.write('run 0 \n')

        script.write('\n')
        script.write('\n')

        # Write out the summary
        script.write('write_dump all custom tempdumpfile id type element mass q x y z vx vy vz fx fy fz modify sort id element ' + ' '.join(opts['atom_sequence']).title() + ' \n')

        script.write('print "-----SUMMARY-----" \n')
        script.write('print "EZFF_TEMP ${ezff_T}" \n')
        script.write('print "EZFF_VOL ${ezff_V}" \n')
        script.write('print "EZFF_ENERGY ${ezff_E}" \n')


        script.close()

        self._write_elastic_displacement_mod()




    def _write_elastic_displacement_mod(self):
        """
        Write-out a complete LAMMPS script file, ``job.scriptfile``, based on job parameters
        """
        opts = self.options
        script = open(self.scriptfile,'a+')

        if opts['pbc']:
            displace_script = open(os.path.join(os.path.abspath(self.path), 'displace.mod'), 'w')

            displace_script.write('clear\nvariable up equal 1.0e-6\nvariable atomjiggle equal 1.0e-5\nvariable cfac equal 1.0e-4\nvariable cunits string GPa\nif "${dir} == 1" then &\n   "variable len0 equal ${lx0}"\nif "${dir} == 2" then &\n   "variable len0 equal ${ly0}"\nif "${dir} == 3" then &\n   "variable len0 equal ${lz0}"\nif "${dir} == 4" then &\n   "variable len0 equal ${lz0}"\nif "${dir} == 5" then &\n   "variable len0 equal ${lz0}"\nif "${dir} == 6" then &\n   "variable len0 equal ${ly0}"\nbox tilt large\nread_restart restart.equil\n')

            ff_include_string = self._include_forcefield(self.options['fftype'])
            displace_script.write(ff_include_string)
            # displace_script.write('pair_style ' + fftype_pairstyle[opts['fftype']] + '\n')
            # displace_script.write('pair_coeff * * ff.lmp ' + ' '.join(opts['atom_sequence']).title() + '\n')
            displace_script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
            displace_script.write('thermo 1 \n')

            displace_script.write('variable delta equal -${up}*${len0}\nvariable deltaxy equal -${up}*xy\nvariable deltaxz equal -${up}*xz\nvariable deltayz equal -${up}*yz\nif "${dir} == 1" then &\n   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"\nif "${dir} == 2" then &\n   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"\nif "${dir} == 3" then &\n   "change_box all z delta 0 ${delta} remap units box"\nif "${dir} == 4" then &\n   "change_box all yz delta ${delta} remap units box"\nif "${dir} == 5" then &\n   "change_box all xz delta ${delta} remap units box"\nif "${dir} == 6" then &\n   "change_box all xy delta ${delta} remap units box"\nminimize 0.0 1.0e-8 1000 100000\nvariable tmp equal pxx\nvariable pxx1 equal ${tmp}\nvariable tmp equal pyy\nvariable pyy1 equal ${tmp}\nvariable tmp equal pzz\nvariable pzz1 equal ${tmp}\nvariable tmp equal pxy\nvariable pxy1 equal ${tmp}\nvariable tmp equal pxz\nvariable pxz1 equal ${tmp}\nvariable tmp equal pyz\nvariable pyz1 equal ${tmp}\nvariable C1neg equal ${d1}\nvariable C2neg equal ${d2}\nvariable C3neg equal ${d3}\nvariable C4neg equal ${d4}\nvariable C5neg equal ${d5}\nvariable C6neg equal ${d6}\nclear\nbox tilt large\nread_restart restart.equil\n')

            ff_include_string = self._include_forcefield(self.options['fftype'])
            displace_script.write(ff_include_string)
            # displace_script.write('pair_style ' + fftype_pairstyle[opts['fftype']] + '\n')
            # displace_script.write('pair_coeff * * ff.lmp ' + ' '.join(opts['atom_sequence']).title() + '\n')
            displace_script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
            displace_script.write('thermo 1 \n')

            displace_script.write('variable delta equal ${up}*${len0}\nvariable deltaxy equal ${up}*xy\nvariable deltaxz equal ${up}*xz\nvariable deltayz equal ${up}*yz\nif "${dir} == 1" then &\n   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"\nif "${dir} == 2" then &\n   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"\nif "${dir} == 3" then &\n   "change_box all z delta 0 ${delta} remap units box"\nif "${dir} == 4" then &\n   "change_box all yz delta ${delta} remap units box"\nif "${dir} == 5" then &\n   "change_box all xz delta ${delta} remap units box"\nif "${dir} == 6" then &\n   "change_box all xy delta ${delta} remap units box"\nminimize 0.0 1.0e-8 1000 100000\nvariable tmp equal pe\nvariable e1 equal ${tmp}\nvariable tmp equal press\nvariable p1 equal ${tmp}\nvariable tmp equal pxx\nvariable pxx1 equal ${tmp}\nvariable tmp equal pyy\nvariable pyy1 equal ${tmp}\nvariable tmp equal pzz\nvariable pzz1 equal ${tmp}\nvariable tmp equal pxy\nvariable pxy1 equal ${tmp}\nvariable tmp equal pxz\nvariable pxz1 equal ${tmp}\nvariable tmp equal pyz\nvariable pyz1 equal ${tmp}\nvariable C1pos equal ${d1}\nvariable C2pos equal ${d2}\nvariable C3pos equal ${d3}\nvariable C4pos equal ${d4}\nvariable C5pos equal ${d5}\nvariable C6pos equal ${d6}\nvariable C1${dir} equal 0.5*(${C1neg}+${C1pos})\nvariable C2${dir} equal 0.5*(${C2neg}+${C2pos})\nvariable C3${dir} equal 0.5*(${C3neg}+${C3pos})\nvariable C4${dir} equal 0.5*(${C4neg}+${C4pos})\nvariable C5${dir} equal 0.5*(${C5neg}+${C5pos})\nvariable C6${dir} equal 0.5*(${C6neg}+${C6pos})\nvariable dir delete\n')

            displace_script.close()

            script.write('\nvariable up equal 1.0e-6\nvariable atomjiggle equal 1.0e-5\nvariable cfac equal 1.0e-4\nvariable cunits string GPa\nvariable tmp equal pxx\nvariable pxx0 equal ${tmp}\nvariable tmp equal pyy\nvariable pyy0 equal ${tmp}\nvariable tmp equal pzz\nvariable pzz0 equal ${tmp}\nvariable tmp equal pyz\nvariable pyz0 equal ${tmp}\nvariable tmp equal pxz\nvariable pxz0 equal ${tmp}\nvariable tmp equal pxy\nvariable pxy0 equal ${tmp}\nvariable tmp equal lx\nvariable lx0 equal ${tmp}\nvariable tmp equal ly\nvariable ly0 equal ${tmp}\nvariable tmp equal lz\nvariable lz0 equal ${tmp}\n# These formulas define the derivatives w.r.t. strain components\n# Constants uses $, variables use v_\nvariable d1 equal -(v_pxx1-${pxx0})/(v_delta/v_len0)*${cfac}\nvariable d2 equal -(v_pyy1-${pyy0})/(v_delta/v_len0)*${cfac}\nvariable d3 equal -(v_pzz1-${pzz0})/(v_delta/v_len0)*${cfac}\nvariable d4 equal -(v_pyz1-${pyz0})/(v_delta/v_len0)*${cfac}\nvariable d5 equal -(v_pxz1-${pxz0})/(v_delta/v_len0)*${cfac}\nvariable d6 equal -(v_pxy1-${pxy0})/(v_delta/v_len0)*${cfac}\n')

            if opts['relax_cell']:
                script.write('unfix FixBoxRelax\n')
            script.write('write_restart restart.equil\n')

            script.write('variable dir equal 1\ninclude displace.mod\nvariable dir equal 2\ninclude displace.mod\nvariable dir equal 3\ninclude displace.mod\nvariable dir equal 4\ninclude displace.mod\nvariable dir equal 5\ninclude displace.mod\nvariable dir equal 6\ninclude displace.mod\nvariable C11all equal ${C11}\nvariable C22all equal ${C22}\nvariable C33all equal ${C33}\nvariable C12all equal 0.5*(${C12}+${C21})\nvariable C13all equal 0.5*(${C13}+${C31})\nvariable C23all equal 0.5*(${C23}+${C32})\nvariable C44all equal ${C44}\nvariable C55all equal ${C55}\nvariable C66all equal ${C66}\nvariable C14all equal 0.5*(${C14}+${C41})\nvariable C15all equal 0.5*(${C15}+${C51})\nvariable C16all equal 0.5*(${C16}+${C61})\nvariable C24all equal 0.5*(${C24}+${C42})\nvariable C25all equal 0.5*(${C25}+${C52})\nvariable C26all equal 0.5*(${C26}+${C62})\nvariable C34all equal 0.5*(${C34}+${C43})\nvariable C35all equal 0.5*(${C35}+${C53})\nvariable C36all equal 0.5*(${C36}+${C63})\nvariable C45all equal 0.5*(${C45}+${C54})\nvariable C46all equal 0.5*(${C46}+${C64})\nvariable C56all equal 0.5*(${C56}+${C65})\nvariable C11cubic equal (${C11all}+${C22all}+${C33all})/3.0\nvariable C12cubic equal (${C12all}+${C13all}+${C23all})/3.0\nvariable C44cubic equal (${C44all}+${C55all}+${C66all})/3.0\nvariable bulkmodulus equal (${C11cubic}+2*${C12cubic})/3.0\nvariable shearmodulus1 equal ${C44cubic}\nvariable shearmodulus2 equal (${C11cubic}-${C12cubic})/2.0\nvariable poissonratio equal 1.0/(1.0+${C11cubic}/${C12cubic})\nprint "EZFF C11 ${C11all} ${cunits}"\nprint "EZFF C22 ${C22all} ${cunits}"\nprint "EZFF C33 ${C33all} ${cunits}"\nprint "EZFF C12 ${C12all} ${cunits}"\nprint "EZFF C13 ${C13all} ${cunits}"\nprint "EZFF C23 ${C23all} ${cunits}"\nprint "EZFF C44 ${C44all} ${cunits}"\nprint "EZFF C55 ${C55all} ${cunits}"\nprint "EZFF C66 ${C66all} ${cunits}"\nprint "EZFF C14 ${C14all} ${cunits}"\nprint "EZFF C15 ${C15all} ${cunits}"\nprint "EZFF C16 ${C16all} ${cunits}"\nprint "EZFF C24 ${C24all} ${cunits}"\nprint "EZFF C25 ${C25all} ${cunits}"\nprint "EZFF C26 ${C26all} ${cunits}"\nprint "EZFF C34 ${C34all} ${cunits}"\nprint "EZFF C35 ${C35all} ${cunits}"\nprint "EZFF C36 ${C36all} ${cunits}"\nprint "EZFF C45 ${C45all} ${cunits}"\nprint "EZFF C46 ${C46all} ${cunits}"\nprint "EZFF C56 ${C56all} ${cunits}"\nprint "EZFF Bulk_Modulus ${bulkmodulus} ${cunits}"\nprint "EZFF Shear_Modulus_1 ${shearmodulus1} ${cunits}"\nprint "EZFF Shear_Modulus_2 ${shearmodulus2} ${cunits}"\nprint "EZFF Poisson_Ratio ${poissonratio}"\n')


        script.close()






    def _write_script_file(self):
        """
        Write-out a complete LAMMPS script file, ``job.scriptfile``, based on job parameters
        """
        opts = self.options
        script = open(self.scriptfile,'w')

        script.write('units metal \n')  # All simulations will be performed in LAMMPS metal units
        script.write('dimension 3 \n')  # All simulations will be 3D
        script.write('atom_style charge \n')

        if opts['pbc']:
            script.write('boundary p p p \n')
        else:
            script.write('boundary fm fm fm \n')

        script.write('read_data input.structure \n')

        ff_include_string = self._include_forcefield(self.options['fftype'])
        script.write(ff_include_string)
        # script.write('pair_style ' + fftype_pairstyle[opts['fftype']] + '\n')
        # script.write('pair_coeff * * ff.lmp ' + ' '.join(opts['atom_sequence']).title() + '\n')

        ## Define a universal thermo style and dump information every step
        script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
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
        script.write('write_dump all custom tempdumpfile id type element mass q x y z vx vy vz fx fy fz modify sort id element ' + ' '.join(opts['atom_sequence']).title() + ' \n')

        script.write('print "-----SUMMARY-----" \n')
        script.write('print "EZFF_TEMP ${ezff_T}" \n')
        script.write('print "EZFF_VOL ${ezff_V}" \n')
        script.write('print "EZFF_ENERGY ${ezff_E}" \n')

        # SCRIPTS TO COMPUTE ELASTIC CONSTANT
        # ADOPTED FROM THE ELASTIC EXAMPLE IN LAMMPS DIRECTORY
        #if opts['pbc']:
        if 1==2:
            displace_script = open(os.path.join(os.path.abspath(self.path), 'displace.mod'), 'w')

            displace_script.write('clear\nvariable up equal 1.0e-6\nvariable atomjiggle equal 1.0e-5\nvariable cfac equal 1.0e-4\nvariable cunits string GPa\nif "${dir} == 1" then &\n   "variable len0 equal ${lx0}"\nif "${dir} == 2" then &\n   "variable len0 equal ${ly0}"\nif "${dir} == 3" then &\n   "variable len0 equal ${lz0}"\nif "${dir} == 4" then &\n   "variable len0 equal ${lz0}"\nif "${dir} == 5" then &\n   "variable len0 equal ${lz0}"\nif "${dir} == 6" then &\n   "variable len0 equal ${ly0}"\nbox tilt large\nread_restart restart.equil\n')

            displace_script.write('pair_style ' + fftype_pairstyle[opts['fftype']] + '\n')
            displace_script.write('pair_coeff * * ff.lmp ' + ' '.join(opts['atom_sequence']).title() + '\n')
            displace_script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
            displace_script.write('thermo 1 \n')

            displace_script.write('variable delta equal -${up}*${len0}\nvariable deltaxy equal -${up}*xy\nvariable deltaxz equal -${up}*xz\nvariable deltayz equal -${up}*yz\nif "${dir} == 1" then &\n   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"\nif "${dir} == 2" then &\n   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"\nif "${dir} == 3" then &\n   "change_box all z delta 0 ${delta} remap units box"\nif "${dir} == 4" then &\n   "change_box all yz delta ${delta} remap units box"\nif "${dir} == 5" then &\n   "change_box all xz delta ${delta} remap units box"\nif "${dir} == 6" then &\n   "change_box all xy delta ${delta} remap units box"\nminimize 0.0 1.0e-8 1000 100000\nvariable tmp equal pxx\nvariable pxx1 equal ${tmp}\nvariable tmp equal pyy\nvariable pyy1 equal ${tmp}\nvariable tmp equal pzz\nvariable pzz1 equal ${tmp}\nvariable tmp equal pxy\nvariable pxy1 equal ${tmp}\nvariable tmp equal pxz\nvariable pxz1 equal ${tmp}\nvariable tmp equal pyz\nvariable pyz1 equal ${tmp}\nvariable C1neg equal ${d1}\nvariable C2neg equal ${d2}\nvariable C3neg equal ${d3}\nvariable C4neg equal ${d4}\nvariable C5neg equal ${d5}\nvariable C6neg equal ${d6}\nclear\nbox tilt large\nread_restart restart.equil\n')

            displace_script.write('pair_style ' + fftype_pairstyle[opts['fftype']] + '\n')
            displace_script.write('pair_coeff * * ff.lmp ' + ' '.join(opts['atom_sequence']).title() + '\n')
            displace_script.write('thermo_style custom step temp pxx pyy pzz pxy pxz pyz pe ke etotal vol xlo xhi ylo yhi zlo zhi xy xz yz press lx ly lz \n')
            displace_script.write('thermo 1 \n')

            displace_script.write('variable delta equal ${up}*${len0}\nvariable deltaxy equal ${up}*xy\nvariable deltaxz equal ${up}*xz\nvariable deltayz equal ${up}*yz\nif "${dir} == 1" then &\n   "change_box all x delta 0 ${delta} xy delta ${deltaxy} xz delta ${deltaxz} remap units box"\nif "${dir} == 2" then &\n   "change_box all y delta 0 ${delta} yz delta ${deltayz} remap units box"\nif "${dir} == 3" then &\n   "change_box all z delta 0 ${delta} remap units box"\nif "${dir} == 4" then &\n   "change_box all yz delta ${delta} remap units box"\nif "${dir} == 5" then &\n   "change_box all xz delta ${delta} remap units box"\nif "${dir} == 6" then &\n   "change_box all xy delta ${delta} remap units box"\nminimize 0.0 1.0e-8 1000 100000\nvariable tmp equal pe\nvariable e1 equal ${tmp}\nvariable tmp equal press\nvariable p1 equal ${tmp}\nvariable tmp equal pxx\nvariable pxx1 equal ${tmp}\nvariable tmp equal pyy\nvariable pyy1 equal ${tmp}\nvariable tmp equal pzz\nvariable pzz1 equal ${tmp}\nvariable tmp equal pxy\nvariable pxy1 equal ${tmp}\nvariable tmp equal pxz\nvariable pxz1 equal ${tmp}\nvariable tmp equal pyz\nvariable pyz1 equal ${tmp}\nvariable C1pos equal ${d1}\nvariable C2pos equal ${d2}\nvariable C3pos equal ${d3}\nvariable C4pos equal ${d4}\nvariable C5pos equal ${d5}\nvariable C6pos equal ${d6}\nvariable C1${dir} equal 0.5*(${C1neg}+${C1pos})\nvariable C2${dir} equal 0.5*(${C2neg}+${C2pos})\nvariable C3${dir} equal 0.5*(${C3neg}+${C3pos})\nvariable C4${dir} equal 0.5*(${C4neg}+${C4pos})\nvariable C5${dir} equal 0.5*(${C5neg}+${C5pos})\nvariable C6${dir} equal 0.5*(${C6neg}+${C6pos})\nvariable dir delete\n')

            displace_script.close()

            script.write('\nvariable up equal 1.0e-6\nvariable atomjiggle equal 1.0e-5\nvariable cfac equal 1.0e-4\nvariable cunits string GPa\nvariable tmp equal pxx\nvariable pxx0 equal ${tmp}\nvariable tmp equal pyy\nvariable pyy0 equal ${tmp}\nvariable tmp equal pzz\nvariable pzz0 equal ${tmp}\nvariable tmp equal pyz\nvariable pyz0 equal ${tmp}\nvariable tmp equal pxz\nvariable pxz0 equal ${tmp}\nvariable tmp equal pxy\nvariable pxy0 equal ${tmp}\nvariable tmp equal lx\nvariable lx0 equal ${tmp}\nvariable tmp equal ly\nvariable ly0 equal ${tmp}\nvariable tmp equal lz\nvariable lz0 equal ${tmp}\n# These formulas define the derivatives w.r.t. strain components\n# Constants uses $, variables use v_\nvariable d1 equal -(v_pxx1-${pxx0})/(v_delta/v_len0)*${cfac}\nvariable d2 equal -(v_pyy1-${pyy0})/(v_delta/v_len0)*${cfac}\nvariable d3 equal -(v_pzz1-${pzz0})/(v_delta/v_len0)*${cfac}\nvariable d4 equal -(v_pyz1-${pyz0})/(v_delta/v_len0)*${cfac}\nvariable d5 equal -(v_pxz1-${pxz0})/(v_delta/v_len0)*${cfac}\nvariable d6 equal -(v_pxy1-${pxy0})/(v_delta/v_len0)*${cfac}\n')

            if opts['relax_cell']:
                script.write('unfix FixBoxRelax\n')
            script.write('write_restart restart.equil\n')

            script.write('variable dir equal 1\ninclude displace.mod\nvariable dir equal 2\ninclude displace.mod\nvariable dir equal 3\ninclude displace.mod\nvariable dir equal 4\ninclude displace.mod\nvariable dir equal 5\ninclude displace.mod\nvariable dir equal 6\ninclude displace.mod\nvariable C11all equal ${C11}\nvariable C22all equal ${C22}\nvariable C33all equal ${C33}\nvariable C12all equal 0.5*(${C12}+${C21})\nvariable C13all equal 0.5*(${C13}+${C31})\nvariable C23all equal 0.5*(${C23}+${C32})\nvariable C44all equal ${C44}\nvariable C55all equal ${C55}\nvariable C66all equal ${C66}\nvariable C14all equal 0.5*(${C14}+${C41})\nvariable C15all equal 0.5*(${C15}+${C51})\nvariable C16all equal 0.5*(${C16}+${C61})\nvariable C24all equal 0.5*(${C24}+${C42})\nvariable C25all equal 0.5*(${C25}+${C52})\nvariable C26all equal 0.5*(${C26}+${C62})\nvariable C34all equal 0.5*(${C34}+${C43})\nvariable C35all equal 0.5*(${C35}+${C53})\nvariable C36all equal 0.5*(${C36}+${C63})\nvariable C45all equal 0.5*(${C45}+${C54})\nvariable C46all equal 0.5*(${C46}+${C64})\nvariable C56all equal 0.5*(${C56}+${C65})\nvariable C11cubic equal (${C11all}+${C22all}+${C33all})/3.0\nvariable C12cubic equal (${C12all}+${C13all}+${C23all})/3.0\nvariable C44cubic equal (${C44all}+${C55all}+${C66all})/3.0\nvariable bulkmodulus equal (${C11cubic}+2*${C12cubic})/3.0\nvariable shearmodulus1 equal ${C44cubic}\nvariable shearmodulus2 equal (${C11cubic}-${C12cubic})/2.0\nvariable poissonratio equal 1.0/(1.0+${C11cubic}/${C12cubic})\nprint "EZFF C11 ${C11all} ${cunits}"\nprint "EZFF C22 ${C22all} ${cunits}"\nprint "EZFF C33 ${C33all} ${cunits}"\nprint "EZFF C12 ${C12all} ${cunits}"\nprint "EZFF C13 ${C13all} ${cunits}"\nprint "EZFF C23 ${C23all} ${cunits}"\nprint "EZFF C44 ${C44all} ${cunits}"\nprint "EZFF C55 ${C55all} ${cunits}"\nprint "EZFF C66 ${C66all} ${cunits}"\nprint "EZFF C14 ${C14all} ${cunits}"\nprint "EZFF C15 ${C15all} ${cunits}"\nprint "EZFF C16 ${C16all} ${cunits}"\nprint "EZFF C24 ${C24all} ${cunits}"\nprint "EZFF C25 ${C25all} ${cunits}"\nprint "EZFF C26 ${C26all} ${cunits}"\nprint "EZFF C34 ${C34all} ${cunits}"\nprint "EZFF C35 ${C35all} ${cunits}"\nprint "EZFF C36 ${C36all} ${cunits}"\nprint "EZFF C45 ${C45all} ${cunits}"\nprint "EZFF C46 ${C46all} ${cunits}"\nprint "EZFF C56 ${C56all} ${cunits}"\nprint "EZFF Bulk_Modulus ${bulkmodulus} ${cunits}"\nprint "EZFF Shear_Modulus_1 ${shearmodulus1} ${cunits}"\nprint "EZFF Shear_Modulus_2 ${shearmodulus2} ${cunits}"\nprint "EZFF Poisson_Ratio ${poissonratio}"\n')


        script.close()



    def cleanup(self):
        """
        Clean-up after the completion of a GULP job. Deletes input, output and forcefields files
        """
        files_to_be_removed = [self.outfile+'.disp', self.outfile+'.dens', self.outfile, self.scriptfile, self.outfile+'.runerror', self.structfile, self.forcefieldfile, self.dumpfile]
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

        raw_energy = _read_energy(self.outfile)

        if self.units == 'metal':
            energy_in_eV = raw_energy
        elif self.units == 'real':
            energy_in_eV = raw_energy / 23.057

        return energy_in_eV

    def read_elastic_moduli(self):
        """
        Read elastic modulus matrix from a completed GULP job

        :returns: 6x6 Elastic modulus matrix in GPa for each input structure, as a list
        """
        return _read_elastic_moduli(self.outfile)

    def read_atomic_charges(self):
        """
        Read atomic charge information from a completed GULP job file

        :returns: xtal.AtTraj object with optimized charge information
        """
        return _read_structure(self.dumpfile) # read_structure already returns xtal object with charges

    def read_structure(self):
        """
        Read converged structure (cell and atomic positions) from the MD job

        :returns: xtal.AtTraj object with (optimized) individual structures as separate snapshots
        """
        return _read_structure(self.dumpfile)



def _read_elastic_moduli(outfilename):
    """
    Read elastic modulus matrix from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: 6x6 Elastic modulus matrix in GPa
    """
    outfile = open(outfilename,'r')
    moduli_array = []
    moduli = np.zeros((6,6))
    while True:
        oneline = outfile.readline()
        if not oneline: # break at EOF
            break
        if 'EZFF C11' in oneline:
            moduli[0,0] = float(oneline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C22
            moduli[1,1] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C33
            moduli[2,2] = float(newline.strip().split()[2]) # in Bars

            newline = outfile.readline() # C12
            moduli[0,1] = float(newline.strip().split()[2]) # in Bars
            moduli[1,0] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C13
            moduli[0,2] = float(newline.strip().split()[2]) # in Bars
            moduli[2,0] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C23
            moduli[1,2] = float(newline.strip().split()[2]) # in Bars
            moduli[2,1] = float(newline.strip().split()[2]) # in Bars

            newline = outfile.readline() # C44
            moduli[3,3] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C55
            moduli[4,4] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C66
            moduli[5,5] = float(newline.strip().split()[2]) # in Bars

            newline = outfile.readline() # C14
            moduli[0,3] = float(newline.strip().split()[2]) # in Bars
            moduli[3,0] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C15
            moduli[0,4] = float(newline.strip().split()[2]) # in Bars
            moduli[4,0] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C16
            moduli[0,5] = float(newline.strip().split()[2]) # in Bars
            moduli[5,0] = float(newline.strip().split()[2]) # in Bars

            newline = outfile.readline() # C24
            moduli[1,3] = float(newline.strip().split()[2]) # in Bars
            moduli[3,1] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C25
            moduli[1,4] = float(newline.strip().split()[2]) # in Bars
            moduli[4,1] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C26
            moduli[1,5] = float(newline.strip().split()[2]) # in Bars
            moduli[5,1] = float(newline.strip().split()[2]) # in Bars

            newline = outfile.readline() # C34
            moduli[2,3] = float(newline.strip().split()[2]) # in Bars
            moduli[3,2] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C35
            moduli[2,4] = float(newline.strip().split()[2]) # in Bars
            moduli[4,2] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C36
            moduli[2,5] = float(newline.strip().split()[2]) # in Bars
            moduli[5,2] = float(newline.strip().split()[2]) # in Bars

            newline = outfile.readline() # C45
            moduli[3,4] = float(newline.strip().split()[2]) # in Bars
            moduli[4,3] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C46
            moduli[3,5] = float(newline.strip().split()[2]) # in Bars
            moduli[5,3] = float(newline.strip().split()[2]) # in Bars
            newline = outfile.readline() # C56
            moduli[4,5] = float(newline.strip().split()[2]) # in Bars
            moduli[5,4] = float(newline.strip().split()[2]) # in Bars

            moduli_array.append(moduli)  # GPa
    outfile.close()
    return moduli_array



def _read_energy(outfilename):
    """
    Read single-point from a completed GULP job

    :param outfilename: Path of the stdout from the GULP job
    :type outfilename: str
    :returns: Energy of the structure in eV
    """
    energy = []
    outfile = open(outfilename, 'r')
    for line in outfile:
        if 'EZFF_ENERGY' in line:
            energy.append(float(line.strip().split()[-1]))
    outfile.close()
    return np.array(energy)



def _read_structure(outfilename):
    """
    Read converged structure (cell and atomic positions) from the MD job

    :param outfilename: Path of file containing stdout of the GULP job
    :type outfilename: str

    :returns: xtal.AtTraj object with (optimized) individual structures as separate snapshots
    """
    relaxed = xtal.AtTraj()
    relaxed.box = np.zeros((3,3))

    # Read number of atoms, box definition and atom coordinates
    outfile = open(outfilename, 'r')
    for line in outfile:
        if 'NUMBER OF ATOMS' in line.strip():
            snapshot = relaxed.create_snapshot(xtal.Snapshot)
            nextline = outfile.readline()
            num_atoms = int(nextline.strip())
        elif 'BOX BOUNDS' in line.strip():
            if 'xy' in line.strip():
                l1 = outfile.readline()
                l2 = outfile.readline()
                l3 = outfile.readline()
                xlo_bound, xhi_bound, xy = list(map(float, l1.strip().split()))
                ylo_bound, yhi_bound, xz = list(map(float, l2.strip().split()))
                zlo_bound, zhi_bound, yz = list(map(float, l3.strip().split()))
                xlo = xlo_bound - np.amin([0.0, xy, xz, xy+xz])
                xhi = xhi_bound - np.amax([0.0, xy, xz, xy+xz])
                ylo = ylo_bound - np.amin([0.0, yz])
                yhi = yhi_bound - np.amax([0.0, yz])
                zlo = zlo_bound
                zhi = zhi_bound
                lx = xhi - xlo
                ly = yhi - ylo
                lz = zhi - zlo
                relaxed.abc = np.array([lx, np.sqrt((ly*ly)+(xy*xy)), np.sqrt((lz*lz)+(xz*xz)+(yz*yz))])
                cosa = ((xy*xz)+(ly*yz))/(relaxed.abc[1]*relaxed.abc[2])
                cosb = xz/relaxed.abc[2]
                cosc = xy/relaxed.abc[1]
                relaxed.ang = np.array([np.arccos(cosa), np.arccos(cosb), np.arccos(cosc)])
                relaxed.abc_to_box()
            else:
                l1 = outfile.readline()
                l2 = outfile.readline()
                l3 = outfile.readline()
                xlo, xhi = list(map(float, l1.strip().split()))
                ylo, yhi = list(map(float, l2.strip().split()))
                zlo, zhi = list(map(float, l3.strip().split()))
                lx = xhi - xlo
                ly = yhi - ylo
                lz = zhi - zlo
                relaxed.box[0][0] = lx
                relaxed.box[1][1] = ly
                relaxed.box[2][2] = lz
        elif 'ATOMS' in line.strip():
            for atomID in range(num_atoms):
                atom_details = outfile.readline().strip().split()
                atom = snapshot.create_atom(xtal.Atom)
                atom.element = atom_details[2].upper()
                atom.charge = float(atom_details[4])
                atom.cart = np.array(list(map(float,atom_details[5:8])))
                atom.vel = np.array(list(map(float,atom_details[8:11])))
                atom.force = np.array(list(map(float,atom_details[11:14])))

    outfile.close()
    return relaxed
