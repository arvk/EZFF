"""Interface to RXMD, the General Utility Lattice Program"""
import os
import xtal
import numpy as np
from ezff.utils import convert_units as convert

class job:
    """
    Class representing a RXMD calculation
    """

    def __init__(self, verbose=False, path='.'):
        """
        :param path: Path where the RXMD job must be run from
        :type path: str

        :param verbose: Print details about the RXMD job
        :type verbose: bool
        """
        if not os.path.isdir(path):
            if verbose:
                print('Path for current job is not valid . Creating a new directory...')
            os.makedirs(path)

        self.path = path
        
        self.outfile = self.path + '/log'
        print(self.outfile)
        self.command = 'rxmd'
        self.forcefield = ''
        self.structure = None

        self.verbose = verbose
        if verbose:
            print('Created a new RXMD job')

    def run(self, command = None, timeout = None):

        """
        Execute rxmd job with user-defined parameters

        :param command: path to RXMD executable
        :type command: str

        :param parallel: Flag for parallel execution
        :type parallel: bool

        :param processors: Number of processors for parallel execution of each RXMD job
        :type processors: int

        :param timeout: RXMD job is automatically killed after ``timeout`` seconds
        :type timeout: int
        """

        if command is None:
            command = self.command
        print(self.outfile)
        system_call_command = command + ' > ' + self.outfile + ' 2> ' + self.outfile + '.runerror'

        if timeout is not None:
            system_call_command = 'timeout ' + str(timeout) + ' ' + system_call_command

        # Create workspace DAT
        makeDir = 'mkdir -p ' + self.path + '/DAT'
        os.system(makeDir)

        # Copy rxmd.in file to path
        copyCommand1 = 'cp rxmd.in ' + self.path  
        copyCommand2 = 'cp rxff_nompi ' + self.path + '/DAT/.'
        

        self.write_script_file()

        os.system(copyCommand1)
        os.system(copyCommand2)        

        if self.verbose:
            print('cd '+ self.path + ' ; ' + system_call_command)

        os.system('cd '+ self.path + ' ; ' + system_call_command)

    def getAtomNames(self,ffieldString, isLG=False):

        atomNames = []
        counter = 0
        #with open(ffieldName, 'r') as ff:
        #ff.readline()
        print('<-----------------------------' + ffieldString[1] + '---------------------------------------->')
        numParams = int(ffieldString[1].split()[0])
        counter += 1
        for i in range(numParams):
            counter += 1

        counter += 1
        numAtomNames = int(ffieldString[counter].split()[0])
        counter += 3

        for i in range(numAtomNames):
            counter += 1
            atomNames.append(ffieldString.split()[0])
            counter += 3

            #if (isLG): ff.readline()
                #print('%d ----> %s' %(i+1, atomNames[-1]))

        print(atomNames)
        return atomNames

    def getBox(self,la,lb,lc,angle1,angle2,angle3):

        H  = np.zeros((3,3))
        Hi = np.zeros((3,3))

        lal = angle1 * (np.pi / 180.0)
        lbe = angle2 * (np.pi / 180.0)
        lga = angle3 * (np.pi / 180.0)

        hh1 = lc * (np.cos(lal) - np.cos(lbe)* np.cos(lga))/ np.sin(lga)
        hh2 = lc * pow(1.0 - pow(np.cos(lal),2) - pow(np.cos(lbe),2) - pow(np.cos(lga),2) + 2 * np.cos(lal) * np.cos(lbe) * np.cos(lga), 0.5)/np.sin(lga)

        H[0,0] = la
        H[1,0] = 0.0
        H[2,0] = 0.0
        H[0,1] = lb * np.cos(lga)
        H[1,1] = lb * np.sin(lga)
        H[2,1] = 0.0
        H[0,2] = lc * np.cos(lbe)
        H[1,2] = hh1
        H[2,2] = hh2

        print('---------------Hmatrix------------------')
        print('%12.6f   %12.6f    %12.6f' %(H[0,0], H[0,1], H[0,2]))
        print('%12.6f   %12.6f    %12.6f' %(H[1,0], H[1,1], H[1,2]))
        print('%12.6f   %12.6f    %12.6f' %(H[2,0], H[2,1], H[2,2]))

        Hi = np.linalg.inv(H)

        print('---------------Hinv--------------------')
        print('%12.6f   %12.6f    %12.6f' %(Hi[0,0], Hi[0,1], Hi[0,2]))
        print('%12.6f   %12.6f    %12.6f' %(Hi[1,0], Hi[1,1], Hi[1,2]))
        print('%12.6f   %12.6f    %12.6f' %(Hi[2,0], Hi[2,1], Hi[2,2]))

        return (H, Hi)

    def write_script_file(self, convert_reaxff=None):

        """
        Prepare input coordinate file to be read by rxmd code
        """

        assert len(self.structure.snaplist[0].atomlist) > 0, "No atoms in structure list"

      
        # To store input file data for rxmd
        #print(self.forcefield)
        #atomNames = self.getAtomNames(self.forcefield)
        atomNames = ['C', 'H', 'O', 'S', 'F', 'Cl', 'N']
        atomLen = len(atomNames)

        natoms = len(self.structure.snaplist[0].atomlist)
        l1, l2, l3, lalpha, lbeta, lgamma = None, None, None, None, None, None
        ctype0, pos, pos0,  itype0, itype1 = [], None, None, None, None

        rr = np.zeros((3,1))
        lbox = np.zeros((3,1))
        obox = np.zeros((3,1))
        H, Hi = np.zeros((3,3)), np.zeros((3,3))

        minX, minY, minZ = 9999, 9999, 9999

        ctype0 = []
        pos    = np.zeros((natoms,3), np.float64)
        pos0   = np.zeros((natoms,3), np.float64)
        itype0 = np.zeros((natoms, ), np.float64)
        itype1 = np.zeros((natoms, ), np.float64)
        
        i=0

        for i, atom in enumerate(self.structure.snaplist[0].atomlist):
            atype = atom.element.title()
            x, y, z = atom.cart[0], atom.cart[1], atom.cart[2]
 
            # Find the minimum coordinate in x, y and z dir and shift the coordinates later such that the origin of box is at 0,0,0
            if x < minX: minX = x
            if y < minY: minY = y
            if z < minZ: minZ = z
 
            pos[i] = np.array([x,y,z])
            #i=i+1

            for j in range(atomLen):
                if atype == atomNames[j]:
                    itype0[i] = j+1
                    break

            # Big floating number to store atom type, atomID, to be read by rxmd code
            itype1[i] = itype0[i] + pow(10,-13)*(i+1) + pow(10,-14)

        # Shift Origin
        pos = pos - np.array([minX, minY, minZ])        
        
        # Find box size

        
        l1, l2, l3 = np.max(pos[:,0]) + 10.0, np.max(pos[:,0]) + 10.0, np.max(pos[:,0]) + 10.0
        lalpha, lbeta, lgamma = 90, 90, 90 
        # Shift all the positions by 0.5
        pos = pos + 5.0

        # Normalize the coordinates by Hi matrix
        H, Hi = self.getBox(l1, l2, l3, lalpha, lbeta, lgamma)

        # Normalized structure is stored in pos0
        pos0 = np.dot(pos, Hi)

        # Wrap backtype0[i] coordinates contained in pos0 if they are out of box
    
        #rmin[0], rmin[1], rmin[2] = np.min(pos0[:,0]), np.min(pos0[:,1]), np.min(pos0[:,2])

        pos0 = pos0 - np.min(pos0,axis=0)   

        # Wrap back more if necessary
        pos0[:,0] = pos0[:,0]%1.0
        pos0[:,1] = pos0[:,1]%1.0
        pos0[:,2] = pos0[:,2]%1.0

        # Shift by a small amount to avoid coordinates at zero
        pos0 = pos0 + pow(10,-9)

        # Check minimum and maximum values of coordinates in pos0
        rmin = np.min(pos0,axis=1)
        rmax = np.max(pos0,axis=1)

        if self.verbose:
            print('rmin ------------> %12.6f  %12.6f  %12.6f' %(rmin[0], rmin[1], rmin[2]))
            print('rmax ------------> %12.6f  %12.6f  %12.6f' %(rmax[0], rmax[1], rmax[2]))

        # Write final coordinates to rxff_nompi to be read by rxmd, also write to geninit1.xyz for verification
        file1 = open('geninit1.xyz', 'w')    
        file2 = open('rxff_nompi', 'w')

        file1.write('%d\n' %(natoms))
        file2.write('1 1 1 1 %d 0\n' %(natoms))

        file1.write('%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n' %(l1,l2,l3,lalpha,lbeta,lgamma))
        file2.write('%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n' %(l1,l2,l3,lalpha,lbeta,lgamma))

        vv = np.zeros((natoms,3))
        rr1 = np.zeros((3,1))
        q = np.zeros((natoms,1))
        qfsp = 0.0
        qfsv = 0.0

        for n in range(natoms):
            rr1 = H.dot(pos0[n])
            dtype = itype1[n]
            gid = int((dtype - int(dtype))*pow(10,13))
            ity = int(dtype)-1
            file1.write('%s %12.6f %12.6f %12.6f %6d\n' %(atomNames[ity], rr1[0], rr1[1], rr1[2], gid))
            file2.write('%12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %16.14f %12.6f %12.6f\n'%(pos0[n][0],pos0[n][1],pos0[n][2],vv[n][0],vv[n][1],vv[n][2],q[n],dtype,qfsp,qfsv))
        
        file1.close()
        file2.close()

    def cleanup(self):
        """
        Clean-up after the completion of a RXMD job. Deletes input, output and forcefields files
        """
        files_to_be_removed = [self.outfile, self.outfile+'.runerror']
        for file in files_to_be_removed:

            if os.path.isfile(file):
                rm_command = 'rm -rf %s ' %file
                os.system(rm_command)
            elif os.path.isfile(self.path+'/'+file):
                rm_command = 'rm -rf %s ' %(self.path + '/' + file)
                os.system(rm_command)

        os.system('rm -rf ' + self.path + '/DAT')
        os.system('rm -rf ' + self.path + '/ffield')
        os.system('rm -rf ' + self.path + '/rxmd.in')
       

    def read_energy(self):
        """
        Read single-point from a completed RXMD job

        :param outfilename: Path of the stdout from the RXMD job
        :type outfilename: str
        :returns: Energy of the structure in eV
        """

        with open(self.outfile,'r') as out:

            for line in out:
                if 'MDstep:' in line:
                    energy_in_eV = float(line.strip().split()[2])*convert.kcal_mol['eV']
        return energy_in_eV
        outfile.close()


    def read_atomic_charges(outfilename):

        """
        Read atomic charge information from a completed rxmd job

        : param outfilename : Path of the file containing information about the rxmd run
        : returns: xtal object with optimized charge information
        """

        structure = xtal.AtTraj()
        snapshot = structure.create_snapshot(xtal.Snapshot)
        #outfile = open(self.path + '/DAT/000000001.xyz', 'r')

        natoms = None
        line, dummyline = None, None
        with open(self.path + '/DAT/000000001.xyz', 'r') as outfile:

            line = outfile.readline()
            natoms = int(line.strip().split()[0])

            dummyline = outfile.readline()
            structure.abc = np.array([ float(dummyline[0]), float(dummyline[1]), float(dummyline[2]) ]) 
            structure.ang = np.array([ float(dummyline[3]), float(dummyline[4]), float(dummyline[5]) ])

            for i in range(natoms):
                charges = outfile.readline()
                charges = charges.strip().split()
                atom = snapshot.create_atom(xtal.Atom)

                if charges[0] == 'C':
                    atom.element = charges[0]
                if charges[0] == 'H':
                    atom.element = charges[0]
                if charges[0] == 'O':
                    atom.element = charges[0]
                if charges[0] == 'N':
                    atom.element = charges[0]
                if charges[0] == 'S':
                    atom.element = charges[0]

                atom.charge = float(charges[4])

        return structure
