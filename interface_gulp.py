"""Interface to GULP, the General Utility Lattice Program"""
import os
import xtal
import numpy as np

class job:
    """A single GULP job"""

    def __init__(self, verbose=False):
        self.scriptfile = 'in.gulp'
        self.outfile = 'out.gulp'
        self.command = 'gulp'
        self.forcefield = ''
        self.structure = None
        self.options = {
            "relax_atoms": False,
            "relax_cell": False,
            "phonon_dispersion": None,
            "phonon_dispersion_from": None,
            "phonon_dispersion_to": None
            }

        if verbose:
            print('Created a new GULP job')

    def run(self, command = None, parallel = False, processors = 1, timeout = None):
        """Execute the GULP job"""
        if command is None:
            command = self.command

        if parallel:
            command = "mpirun -np " + str(processors) + " " + command

        system_call_command = command + ' < ' + self.scriptfile + ' > ' + self.outfile

        if timeout is not None:
            system_call_command = 'timeout ' + str(timeout) + ' ' + system_call_command

        print(system_call_command)
        os.system(system_call_command)


    def read_atomic_structure(self,structure_file):
        structure = xtal.AtTraj(verbose=False)

        if ('POSCAR' in structure_file) or ('CONTCAR' in structure_file):
            structure.read_snapshot_vasp(structure_file)

        return structure


    def write_script_file(self):
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

        if header_line == '':
            header_line = 'single '

        header_line += 'comp '
        script.write(header_line + '\n')

        script.write('\n')

        script.write('vectors\n')
        script.write(np.array_str(self.structure.box).replace('[','').replace(']','') + '\n')
        script.write('Fractional\n')
        for atom in self.structure.snaplist[0].atomlist:
            positions = atom.element.title() + ' core '
            positions += np.array_str(atom.fract).replace('[','').replace(']','')
            positions += ' 0.0   1.0   0.0   1 1 1 \n'
            script.write(positions)
        script.write('\n')

        with open(self.forcefield,'r') as forcefield_file:
            forcefield = forcefield_file.read()

        for line in forcefield.split('\n'):
            script.write(' '.join(line.split()) + '\n')

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
        os.remove(self.outfile+'.disp')
        os.remove(self.outfile+'.dens')
        os.remove(self.outfile)
        os.remove(self.scriptfile)
