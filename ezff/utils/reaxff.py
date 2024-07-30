import numpy as np
import itertools
import math



class reax_forcefield:
    """
    ReaxFF forcefield class. Used for generating ReaxFF templates
    """
    def __init__(self, filename = None, filestring = None, template = 'ff.template.generated', ranges = 'param_ranges', bo_threshold = 1e-8):
        """
        :param filename: ReaxFF forcefield filename
        :type filename: str

        :param filestring: ReaxFF forcefield filestring
        :type filestring: str

        :param template: ReaxFF forcefield template filename
        :type template: str

        :param ranges: File containing the lower and upper bounds for decision variables
        :type ranges: str
        """
        self.params_write = []
        self.template = template
        self.ranges = ranges
        self.bo_threshold = bo_threshold
        if not filename is None:
            self.read_forcefield_from_file(filename)
        elif not filestring is None:
            self.read_forcefield_from_string(filestring)



    def read_forcefield_from_file(self,filename):
        """
        Read ReaxFF forcefield from external file

        :param filename: ReaxFF forcefield filename
        :type filename: str
        """
        fffile = open(filename,'r')
        ff = fffile.read()
        fffile.close()

        # Parse forcefield from read-in string
        self.read_forcefield_from_string(ff)
        return



    def read_forcefield_from_string(self,filestring):
        """
        Read ReaxFF forcefield from a given forcefield string

        :param filestring: ReaxFF forcefield string
        :type filestring: str
        """
        list_of_strings = [line+'\n' for line in filestring.split('\n')]

        self.full = list_of_strings
        #print(self.full)
        # Split forcefield
        self._split_forcefield()
        return



    def _split_forcefield(self):
        """
        Split ReaxFF forcefield into sections corresponding to general, one-body, two-body, three-body, four-body, offdiagonal and H-bond sections
        """
        header, general, onebody, twobody, offdiagonal, threebody, fourbody, hbond = [], [], [], [], [], [], [], []
        counter = 0
        ff = self.full

        # Read HEADER line
        header = ff[0]
        counter += 1

        num_general = int(ff[counter].strip().split()[0])
        general_string = ff[counter:counter+num_general+1] # one for the header another for the number of parameters
        for line in general_string:
            general.append(line.strip().split())
        counter += (num_general + 1)

        num_onebody = int(ff[counter].strip().split()[0])
        onebody_string = ff[counter:counter+(num_onebody*4)+4] # one for the header another for the number of parameters
        for line in onebody_string:
            onebody.append(line.strip().split())
        counter += ((num_onebody*4) + 4)

        num_twobody = int(ff[counter].strip().split()[0])
        twobody_string = ff[counter:counter+(num_twobody*2)+2] # one for the header another for the number of parameters
        for line in twobody_string:
            twobody.append(line.strip().split())
        counter += ((num_twobody*2) + 2)

        num_offdiagonal = int(ff[counter].strip().split()[0])
        offdiagonal_string = ff[counter:counter+(num_offdiagonal*1)+1] # one for the header another for the number of parameters
        for line in offdiagonal_string:
            offdiagonal.append(line.strip().split())
        counter += ((num_offdiagonal*1) + 1)

        num_threebody = int(ff[counter].strip().split()[0])
        threebody_string = ff[counter:counter+(num_threebody*1)+1] # one for the header another for the number of parameters
        for line in threebody_string:
            threebody.append(line.strip().split())
        counter += ((num_threebody*1) + 1)

        num_fourbody = int(ff[counter].strip().split()[0])
        fourbody_string = ff[counter:counter+(num_fourbody*1)+1] # one for the header another for the number of parameters
        for line in fourbody_string:
            fourbody.append(line.strip().split())
        counter += ((num_fourbody*1) + 1)

        num_hbond = int(ff[counter].strip().split()[0])
        hbond_string = ff[counter:counter+(num_hbond*1)+1] # one for the header another for the number of parameters
        for line in hbond_string:
            hbond.append(line.strip().split())
        counter += ((num_hbond*1) + 1)

        self.header = header
        self.general = general
        self.onebody = onebody
        self.twobody = twobody
        self.offdiagonal = offdiagonal
        self.threebody = threebody
        self.fourbody = fourbody
        self.hbond = hbond



    def _get_element_number(self,element):
        """
        Get the numerical index of an element in the ReaxFF forcefield file

        :param element: Chemical symbol for the element
        :type element: str
        """
        new_onebody = self.onebody[4::4]
        for i in range(len(new_onebody)):
            if new_onebody[i][0].lower().upper() == element.lower().upper():
                return i+1
        return 0



    def _template_qeq(self, e1, bounds):
       """
       Generate decision variable for electrostatic energy equation for a particular element

       :param e1: Chemical symbol for element 1
       :type e1: str

       :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
       :type bounds: float

       """
       ie1 = self._get_element_number(e1)
       if ie1 == 0: return

       # gamma, chi and eta
       for index, line in enumerate(self.onebody[4::4]):
           if (line[0] == e1):
               break
       line_number = 3 + (4*index) + 1

       gamma = float(self.onebody[line_number][7-1]) #  6th term, -1 for 0 indexing
       chi   = float(self.onebody[line_number+1][14-8-1]) # 14th term, -8 for previous line, -1 for 0 indexing
       eta   = float(self.onebody[line_number+1][15-8-1]) # 15th term, -8 for previous line, -1 for 0 indexing

       # gamma
       self.onebody[line_number][7-1] = '<<gam_'+e1+'>>'
       delta = bounds * np.absolute(gamma)
       self.params_write.append(['gam_'+e1, str(gamma-delta), str(gamma+delta)])

       # chi
       self.onebody[line_number+1][14-8-1] = '<<chi_'+e1+'>>'
       delta = bounds * np.absolute(chi)
       self.params_write.append(['chi_'+e1, str(chi-delta), str(chi+delta)])

       # eta
       self.onebody[line_number+1][15-8-1] = '<<eta_'+e1+'>>'
       delta = bounds * np.absolute(eta)
       self.params_write.append(['eta_'+e1, str(eta-delta), str(eta+delta)])

       return



    def _template_onebody(self, e1, bounds):
       """
       Generate decision variable for electrostatic energy equation for a particular element

       :param e1: Chemical symbol for element 1
       :type e1: str

       :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
       :type bounds: float

       """
       ie1 = self._get_element_number(e1)
       if ie1 == 0: return

       # gamma, chi and eta
       for index, line in enumerate(self.onebody[4::4]):
           if (line[0] == e1):
               break
       line_number = 3 + (4*index) + 1

       # Line 1
       r0_sigma = float(self.onebody[line_number][1+1-1]) #  1st term (+1 for element name), -1 for 0 indexing
       rvdW = float(self.onebody[line_number][4+1-1]) #  4th term (+1 for element name), -1 for 0 indexing
       EvdW = float(self.onebody[line_number][5+1-1]) #  5th term (+1 for element name), -1 for 0 indexing
       r0_pi = float(self.onebody[line_number][7+1-1]) #  7th term (+1 for element name), -1 for 0 indexing

       # Line 2
       alfa = float(self.onebody[line_number+1][1-1]) #  1st term, -1 for 0 indexing
       gamma_w_inv = float(self.onebody[line_number+1][2-1]) #  2nd term, -1 for 0 indexing
       Eunder = float(self.onebody[line_number+1][4-1]) #  4th term, -1 for 0 indexing
       Eover = float(self.onebody[line_number+1][5-1]) #  5th term, -1 for 0 indexing

       # Line 3
       r0_pipi = float(self.onebody[line_number+2][1-1]) #  1st term, -1 for 0 indexing
       Elp = float(self.onebody[line_number+2][2-1]) #  2nd term, -1 for 0 indexing
       pboc4 = float(self.onebody[line_number+2][4-1]) #  4th term, -1 for 0 indexing
       pboc3 = float(self.onebody[line_number+2][5-1]) #  5th term, -1 for 0 indexing
       pboc5 = float(self.onebody[line_number+2][6-1]) #  6th term, -1 for 0 indexing

       # Line 4
       povun2 = float(self.onebody[line_number+3][1-1]) #  1st term, -1 for 0 indexing
       pval3 = float(self.onebody[line_number+3][2-1]) #  2nd term, -1 for 0 indexing
       pval5 = float(self.onebody[line_number+3][5-1]) #  5th term, -1 for 0 indexing


       # r0_sigma
       self.onebody[line_number][1+1-1] = '<<r0_sigma_'+e1+'>>'
       delta = bounds * np.absolute(r0_sigma)
       self.params_write.append(['r0_sigma_'+e1, str(r0_sigma-delta), str(r0_sigma+delta)])

       # rvdW
       self.onebody[line_number][4+1-1] = '<<rvdW_'+e1+'>>'
       delta = bounds * np.absolute(rvdW)
       self.params_write.append(['rvdW_'+e1, str(rvdW-delta), str(rvdW+delta)])

       # EvdW
       self.onebody[line_number][5+1-1] = '<<EvdW_'+e1+'>>'
       delta = bounds * np.absolute(EvdW)
       self.params_write.append(['EvdW_'+e1, str(EvdW-delta), str(EvdW+delta)])

       # r0_pi
       self.onebody[line_number][7+1-1] = '<<r0_pi_'+e1+'>>'
       delta = bounds * np.absolute(r0_pi)
       self.params_write.append(['r0_pi_'+e1, str(r0_pi-delta), str(r0_pi+delta)])

       # alfa
       self.onebody[line_number+1][1-1] = '<<alfa_'+e1+'>>'
       delta = bounds * np.absolute(alfa)
       self.params_write.append(['alfa_'+e1, str(alfa-delta), str(alfa+delta)])

       # gamma_w_inv
       self.onebody[line_number+1][2-1] = '<<gamma_w_inv_'+e1+'>>'
       delta = bounds * np.absolute(gamma_w_inv)
       self.params_write.append(['gamma_w_inv_'+e1, str(gamma_w_inv-delta), str(gamma_w_inv+delta)])

       # Eunder
       self.onebody[line_number+1][4-1] = '<<Eunder_'+e1+'>>'
       delta = bounds * np.absolute(Eunder)
       self.params_write.append(['Eunder_'+e1, str(Eunder-delta), str(Eunder+delta)])

       # Eover
       self.onebody[line_number+1][5-1] = '<<Eover_'+e1+'>>'
       delta = bounds * np.absolute(Eover)
       self.params_write.append(['Eover_'+e1, str(Eover-delta), str(Eover+delta)])

       # r0_pipi
       self.onebody[line_number+2][1-1] = '<<r0_pipi_'+e1+'>>'
       delta = bounds * np.absolute(r0_pipi)
       self.params_write.append(['r0_pipi_'+e1, str(r0_pipi-delta), str(r0_pipi+delta)])

       # Elp
       if np.absolute(Elp) > 0.001:
           self.onebody[line_number+2][2-1] = '<<Elp_'+e1+'>>'
           delta = bounds * np.absolute(Elp)
           self.params_write.append(['Elp_'+e1, str(Elp-delta), str(Elp+delta)])

       # pboc4
       if np.absolute(pboc4) > 0.001:
           self.onebody[line_number+2][4-1] = '<<pboc4_'+e1+'>>'
           delta = bounds * np.absolute(pboc4)
           self.params_write.append(['pboc4_'+e1, str(pboc4-delta), str(pboc4+delta)])

       # pboc3
       if np.absolute(pboc3) > 0.001:
           self.onebody[line_number+2][5-1] = '<<pboc3_'+e1+'>>'
           delta = bounds * np.absolute(pboc3)
           self.params_write.append(['pboc3_'+e1, str(pboc3-delta), str(pboc3+delta)])

       # pboc5
       if np.absolute(pboc5) > 0.001:
           self.onebody[line_number+2][6-1] = '<<pboc5_'+e1+'>>'
           delta = bounds * np.absolute(pboc5)
           self.params_write.append(['pboc5_'+e1, str(pboc5-delta), str(pboc5+delta)])

       # povun2
       self.onebody[line_number+3][1-1] = '<<povun2_'+e1+'>>'
       delta = bounds * np.absolute(povun2)
       self.params_write.append(['povun2_'+e1, str(povun2-delta), str(povun2+delta)])

       # pval3
       self.onebody[line_number+3][2-1] = '<<pval3_'+e1+'>>'
       delta = bounds * np.absolute(pval3)
       self.params_write.append(['pval3_'+e1, str(pval3-delta), str(pval3+delta)])

       # pval5
       self.onebody[line_number+3][5-1] = '<<pval5_'+e1+'>>'
       delta = bounds * np.absolute(pval5)
       self.params_write.append(['pval5_'+e1, str(pval5-delta), str(pval5+delta)])

       return




    def _template_bond_order(self, e1, e2, double_bond = False, triple_bond = False, bounds = 0.1):
        """
        Generate decision variables in the bond-order equation for bonds between two elements

        :param e1: Chemical symbol for element 1
        :type e1: str

        :param e2: Chemical symbol for element 2
        :type e2: str

        :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        :type bounds: float

        :param double_bond: Flag for the presence of a double-bond between elements e1 and e2
        :type double_bond: bool

        :param triple_bond: Flag for the presence of a triple-bond between elements e1 and e2
        :type triple_bond: bool

        """
        ie1, ie2 = self._get_element_number(e1), self._get_element_number(e2)

        #-------------------#
        #--- SINGLE BOND ---#
        #-------------------#

        # PBO1 and PBO2
        for index, line in enumerate(self.twobody[2::2]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (2 + (2*index) + 1)

        #PBO1
        PBO1 = float(self.twobody[line_number][13-8-1]) # 13th term, -8 for previous line, -1 for 0 indexing
        delta = bounds * np.absolute(PBO1)
        self.twobody[line_number][13-8-1] = '<<PBO1_'+e1+'_'+e2+'>>'
        self.params_write.append(['PBO1_'+e1+'_'+e2, str(PBO1-delta), str(PBO1+delta)])

        #PBO2
        PBO2 = float(self.twobody[line_number][14-8-1]) # 14th term, -8 for previous line, -1 for 0 indexing
        delta = bounds * np.absolute(PBO2)
        self.twobody[line_number][14-8-1] = '<<PBO2_'+e1+'_'+e2+'>>'
        self.params_write.append(['PBO2_'+e1+'_'+e2, str(PBO2-delta), str(PBO2+delta)])

        # ro_sigma
        for index, line in enumerate(self.offdiagonal[1:]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (1 + (1*index))

        # ro_sigma
        ro_sigma = float(self.offdiagonal[line_number][4+2-1])  # 4th term, +2 for atom indices, -1 for 0 indexing
        delta = bounds * np.absolute(ro_sigma)
        self.offdiagonal[line_number][4+2-1] = '<<ro_sigma_'+e1+'_'+e2+'>>'
        self.params_write.append(['ro_sigma_'+e1+'_'+e2, str(ro_sigma-delta), str(ro_sigma+delta)])



        #-------------------#
        #--- DOUBLE BOND ---#
        #-------------------#
        if double_bond:
            # PBO1 and PBO2
            for index, line in enumerate(self.twobody[2::2]):
                if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                    break
            line_number = (2 + (2*index) + 1)

            #PBO3
            PBO3 = float(self.twobody[line_number][10-8-1]) # 10th term, -8 for previous line, -1 for 0 indexing
            delta = bounds * np.absolute(PBO3)
            self.twobody[line_number][10-8-1] = '<<PBO3_'+e1+'_'+e2+'>>'
            self.params_write.append(['PBO3_'+e1+'_'+e2, str(PBO3-delta), str(PBO3+delta)])

            #PBO4
            PBO4 = float(self.twobody[line_number][11-8-1]) # 11th term, -8 for previous line, -1 for 0 indexing
            delta = bounds * np.absolute(PBO4)
            self.twobody[line_number][11-8-1] = '<<PBO4_'+e1+'_'+e2+'>>'
            self.params_write.append(['PBO4_'+e1+'_'+e2, str(PBO4-delta), str(PBO4+delta)])

            # ro_pi
            for index, line in enumerate(self.offdiagonal[1:]):
                if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                    break
            line_number = (1 + (1*index))

            # ro_pi
            ro_pi = float(self.offdiagonal[line_number][5+2-1])  # 4th term, +2 for atom indices, -1 for 0 indexing
            delta = bounds * np.absolute(ro_pi)
            self.offdiagonal[line_number][5+2-1] = '<<ro_pi_'+e1+'_'+e2+'>>'
            self.params_write.append(['ro_pi_'+e1+'_'+e2, str(ro_pi-delta), str(ro_pi+delta)])


        #-------------------#
        #--- TRIPLE BOND ---#
        #-------------------#
        if triple_bond:
            # PBO5 and PBO6
            for index, line in enumerate(self.twobody[2::2]):
                if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                    break
            line_number = (2 + (2*index))

            #PBO5
            PBO5 = float(self.twobody[line_number][5+2-1]) # 5th term, +2 for atom indices, -1 for 0 indexing
            delta = bounds * np.absolute(PBO5)
            self.twobody[line_number][5+2-1] = '<<PBO5_'+e1+'_'+e2+'>>'
            self.params_write.append(['PBO5_'+e1+'_'+e2, str(PBO5-delta), str(PBO5+delta)])

            #PBO6
            PBO6 = float(self.twobody[line_number][8+2-1]) # 8th term, +2 for atom indices, -1 for 0 indexing
            delta = bounds * np.absolute(PBO6)
            self.twobody[line_number][8+2-1] = '<<PBO6_'+e1+'_'+e2+'>>'
            self.params_write.append(['PBO6_'+e1+'_'+e2, str(PBO6-delta), str(PBO6+delta)])

            # ro_pipi
            for index, line in enumerate(self.offdiagonal[1:]):
                if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                    break
            line_number = (1 + (1*index))

            # ro_pipi
            ro_pipi = float(self.offdiagonal[line_number][6+2-1]) # 6th term, +2 for atom indices, -1 for 0 indexing
            delta = bounds * np.absolute(ro_pipi)
            self.offdiagonal[line_number][6+2-1] = '<<ro_pipi_'+e1+'_'+e2+'>>'
            self.params_write.append(['ro_pipi_'+e1+'_'+e2, str(ro_pipi-delta), str(ro_pipi+delta)])

        return



    def _template_bond_energy_attractive(self, e1, e2, double_bond = False, triple_bond = False, bounds = 0.1):
        """
        Generate decision variables related to the two-body attractive term

        :param e1: Chemical symbol for element 1
        :type e1: str

        :param e2: Chemical symbol for element 2
        :type e2: str

        :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        :type bounds: float

        :param double_bond: Flag for the presence of a double-bond between elements e1 and e2
        :type double_bond: bool

        :param triple_bond: Flag for the presence of a triple-bond between elements e1 and e2
        :type triple_bond: bool
        """
        ie1, ie2 = self._get_element_number(e1), self._get_element_number(e2)

        #-------------------#
        #--- SINGLE BOND ---#
        #-------------------#

        # De_sigma
        for index, line in enumerate(self.twobody[2::2]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (2 + (2*index))

        #De_sigma
        De_sigma = float(self.twobody[line_number][1+2-1]) # 1st term, +2 for atom indices, -1 for 0 indexing
        delta = bounds * np.absolute(De_sigma)
        self.twobody[line_number][1+2-1] = '<<De_sigma_'+e1+'_'+e2+'>>'
        self.params_write.append(['De_sigma_'+e1+'_'+e2, str(De_sigma-delta), str(De_sigma+delta)])

        #PBE1
        PBE1 = float(self.twobody[line_number][4+2-1]) # 4th term, +2 for atom indices, -1 for 0 indexing
        delta = bounds * np.absolute(PBE1)
        self.twobody[line_number][4+2-1] = '<<PBE1_'+e1+'_'+e2+'>>'
        self.params_write.append(['PBE1_'+e1+'_'+e2, str(PBE1-delta), str(PBE1+delta)])

        line_number = (2 + (2*index) + 1)  # PBE2 is on the next line

        #PBE2
        PBE2 = float(self.twobody[line_number][9-8-1]) # 9th term, -8 for previous line, -1 for 0 indexing
        delta = bounds * np.absolute(PBE2)
        self.twobody[line_number][9-8-1] = '<<PBE2_'+e1+'_'+e2+'>>'
        self.params_write.append(['PBE2_'+e1+'_'+e2, str(PBE2-delta), str(PBE2+delta)])

        #-------------------#
        #--- DOUBLE BOND ---#
        #-------------------#
        # De_pi
        for index, line in enumerate(self.twobody[2::2]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (2 + (2*index))
        De_pi = float(self.twobody[line_number][2+2-1]) # 2nd term, +2 for atom indices, -1 for 0 indexing

        if double_bond:
            delta = bounds * np.absolute(De_pi)
            self.twobody[line_number][2+2-1] = '<<De_pi_'+e1+'_'+e2+'>>'
            self.params_write.append(['De_pi_'+e1+'_'+e2, str(De_pi-delta), str(De_pi+delta)])
        else:
            if De_pi != 0.0:
                print('Double bond parameters for ' + e1 + '-' + e2 + ' bond will not be optimized. Current non-zero values in the template forcefield will be retained.')

        #-------------------#
        #--- TRIPLE BOND ---#
        #-------------------#
        # De_pipi
        for index, line in enumerate(self.twobody[2::2]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (2 + (2*index))
        De_pipi = float(self.twobody[line_number][3+2-1]) # 3rd term, +2 for atom indices, -1 for 0 indexing

        if triple_bond:
            delta = bounds * np.absolute(De_pipi)
            self.twobody[line_number][3+2-1] = '<<De_pipi_'+e1+'_'+e2+'>>'
            self.params_write.append(['De_pipi_'+e1+'_'+e2, str(De_pipi-delta), str(De_pipi+delta)])
        else:
            if De_pipi != 0.0:
                print('Triple bond parameters for ' + e1 + '-' + e2 + ' bond will not be optimized. Current non-zero values in the template forcefield will be retained.')





    def _template_bond_energy_vdW(self, e1, e2, bounds = 0.1):
        """
        Generate decision variables related to the two-body repulsive (i.e. van der Waals) term

        :param e1: Chemical symbol for element 1
        :type e1: str

        :param e2: Chemical symbol for element 2
        :type e2: str

        :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        :type bounds: float
        """
        ie1, ie2 = self._get_element_number(e1), self._get_element_number(e2)

        # Dij, rvdWm alpha_ij in off-diagonal
        for index, line in enumerate(self.offdiagonal[1:]):
            if (int(line[0]) == ie1 and int(line[1]) == ie2) or (int(line[0]) == ie2 and int(line[1]) == ie1):
                break
        line_number = (1 + (1*index))

        # Dij
        Dij = float(self.offdiagonal[line_number][1+2-1]) # 1st term, +2 for atom indices, -1 for 0 indexing
        delta = bounds * np.absolute(Dij)
        self.offdiagonal[line_number][1+2-1] = '<<Dij_'+e1+'_'+e2+'>>'
        self.params_write.append(['Dij_'+e1+'_'+e2, str(Dij-delta), str(Dij+delta)])

        # rvdW
        rvdW = float(self.offdiagonal[line_number][2+2-1]) # 2nd term, +2 for atom indices, -1 for 0 indexing
        delta = bounds * np.absolute(rvdW)
        self.offdiagonal[line_number][2+2-1] = '<<rvdW_'+e1+'_'+e2+'>>'
        self.params_write.append(['rvdW_'+e1+'_'+e2, str(rvdW-delta), str(rvdW+delta)])

        # alpha_ij
        alpha_ij = float(self.offdiagonal[line_number][3+2-1]) # 3rd term, +2 for atom indices, -1 for 0 indexing
        delta = bounds * np.absolute(alpha_ij)
        self.offdiagonal[line_number][3+2-1] = '<<alpha_ij_'+e1+'_'+e2+'>>'
        self.params_write.append(['alpha_ij_'+e1+'_'+e2, str(alpha_ij-delta), str(alpha_ij+delta)])

        #     # Pvdw1 in general parameters
        #     line_number = 1 + 29  # 29th parameter, +1 for header line
        #     P_vdW1 = float(self.general[line_number][0])
        #     delta = bounds * np.absolute(P_vdW1)
        #     self.general[line_number][0] = '<<PvdW>>'
        #     self.params_write.append(['PvdW', str(P_vdW1-delta), str(P_vdW1+delta)])








    def _template_threebody_energy(self, e1, e2, e3, bounds = 0.1):
        """
        Generate decision variables related to the three-body angle term

        :param e1: Chemical symbol for element 1
        :type e1: str

        :param e2: Chemical symbol for element 2
        :type e2: str

        :param e3: Chemical symbol for element 3
        :type e3: str

        :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        :type bounds: float
        """
        # theta0, Pval1, Pval2 in threebody
        for triplet in list(set(list(itertools.permutations([e1,e2,e3])))):
            ie1 = self._get_element_number(triplet[0])
            ie2 = self._get_element_number(triplet[1])
            ie3 = self._get_element_number(triplet[2])
            for index, line in enumerate(self.threebody[1:]):
                if int(line[0]) == ie1 and int(line[1]) == ie2 and int(line[2]) == ie3:
                    line_number = (1 + (1*index))

                    theta0 = float(self.threebody[line_number][1+3-1]) # 1st term, +3 for atom indices, -1 for 0 indexing
                    delta = bounds * np.absolute(theta0)
                    self.threebody[line_number][1+3-1] = '<<theta0_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2]+'>>'
                    self.params_write.append(['theta0_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2], str(theta0-delta), str(theta0+delta)])

                    Pval1 = float(self.threebody[line_number][2+3-1]) # 2nd term, +3 for atom indices, -1 for 0 indexing
                    delta = bounds * np.absolute(Pval1)
                    self.threebody[line_number][2+3-1] = '<<Pval1_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2]+'>>'
                    self.params_write.append(['Pval1_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2], str(Pval1-delta), str(Pval1+delta)])

                    Pval2 = float(self.threebody[line_number][3+3-1]) # 3rd term, +3 for atom indices, -1 for 0 indexing
                    delta = bounds * np.absolute(Pval2)
                    self.threebody[line_number][3+3-1] = '<<Pval2_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2]+'>>'
                    self.params_write.append(['Pval2_'+triplet[0]+'_'+triplet[1]+'_'+triplet[2], str(Pval2-delta), str(Pval2+delta)])




    def _template_fourbody_energy(self, e1, e2, e3, e4, bounds = 0.1):
        """
        Generate decision variables related to the four-body dihedral term

        :param e1: Chemical symbol for element 1
        :type e1: str

        :param e2: Chemical symbol for element 2
        :type e2: str

        :param e3: Chemical symbol for element 3
        :type e3: str

        :param e4: Chemical symbol for element 4
        :type e4: str

        :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        :type bounds: float
        """
        # V1, V2, V3, Ptor1 in fourbody
        for quartet in list(set(list(itertools.permutations([e1,e2,e3,e4])))):
            ie1 = self._get_element_number(quartet[0])
            ie2 = self._get_element_number(quartet[1])
            ie3 = self._get_element_number(quartet[2])
            ie4 = self._get_element_number(quartet[3])

            for index, line in enumerate(self.fourbody[1:]):
                if int(line[0]) == ie1 and int(line[1]) == ie2 and int(line[2]) == ie3 and int(line[3]) == ie4:
                    line_number = (1 + (1*index))

                    V1 = float(self.fourbody[line_number][1+4-1]) # 1st term, +3 for atom indices, -1 for 0 indexing
                    delta = bounds * np.absolute(V1)
                    self.fourbody[line_number][1+4-1] = '<<V1_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3]+'>>'
                    self.params_write.append(['V1_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3], str(V1-delta), str(V1+delta)])

                    V2 = float(self.fourbody[line_number][2+4-1]) # 2nd term, +3 for atom indices, -1 for 0 indexing
                    delta = bounds * np.absolute(V2)
                    self.fourbody[line_number][2+4-1] = '<<V2_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3]+'>>'
                    self.params_write.append(['V2_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3], str(V2-delta), str(V2+delta)])

                    V3 = float(self.fourbody[line_number][3+4-1]) # 3rd term, +3 for atom indices, -1 for 0 indexing
                    delta = bounds * np.absolute(V3)
                    self.fourbody[line_number][3+4-1] = '<<V3_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3]+'>>'
                    self.params_write.append(['V3_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3], str(V3-delta), str(V3+delta)])

                    Ptor1 = float(self.fourbody[line_number][4+4-1]) # 4th term, +3 for atom indices, -1 for 0 indexing
                    delta = bounds * np.absolute(Ptor1)
                    self.fourbody[line_number][4+4-1] = '<<Ptor1_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3]+'>>'
                    self.params_write.append(['Ptor1_'+quartet[0]+'_'+quartet[1]+'_'+quartet[2]+'_'+quartet[3], str(Ptor1-delta), str(Ptor1+delta)])



    def generate_templates(self):
        """
        Function to write-out the current modified forcefield sections into a forcefield template file
        """
        with open(self.ranges, 'w') as ranges_file:
            for parameter in self.params_write:
                ranges_file.write("%18s %10.4f %10.4f \n" % (parameter[0], float(parameter[1]), float(parameter[2])))
                #ranges_file.write(' '.join(parameter) + '\n')

        with open(self.template,'w') as template:
            template.write(self.header)
            for line in self.general:
                template.write(' '.join(line)+'\n')
            for line in self.onebody:
                template.write(' '.join(line)+'\n')
            for line in self.twobody:
                template.write(' '.join(line)+'\n')
            for line in self.offdiagonal:
                template.write(' '.join(line)+'\n')
            for line in self.threebody:
                template.write(' '.join(line)+'\n')
            for line in self.fourbody:
                template.write(' '.join(line)+'\n')
            for line in self.hbond:
                template.write(' '.join(line)+'\n')


    def make_template_qeq(self, e1, bounds=0.1):
        """
        Function to generate decision variable for Charge Equilibration (QEq) terms

        : param e1 : Chemical symbol for element 1
        : type e1  : str

        : param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        : type bounds: float

        """
        # GET ONE_BODY_PARAMETERS SPECIFIC TO QEQ
        self._template_qeq(e1,bounds)
        return


    def make_template_onebody(self, e1, bounds=0.1):
        """
        Function to generate decision variables for all one-body terms (i.e. atomic radii, bond_order correction, under/overcoordination)

        : param e1 : Chemical symbol for element 1
        : type e1  : str

        : param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        : type bounds: float

        """
        # GET ONE_BODY_PARAMETERS EXCEPT QEQ
        self._template_onebody(e1,bounds)
        return


    def make_template_twobody(self, e1, e2, double_bond = False, triple_bond = False, bounds = 0.1):
        """
        Function to generate decision variables for all two-body terms (i.e. bond-order, attractive and vdW) between two given elements

        :param e1: Chemical symbol for element 1
        :type e1: str

        :param e2: Chemical symbol for element 2
        :type e2: str

        :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        :type bounds: float

        :param double_bond: Flag for the presence of a double-bond between elements e1 and e2
        :type double_bond: bool

        :param triple_bond: Flag for the presence of a triple-bond between elements e1 and e2
        :type triple_bond: bool
        """

        # Modify One Body Parameters for each unique element
        unique_elements = list(set([e1,e2]))
        for element in unique_elements:
            self._template_onebody(element, bounds=bounds)

        # GET BOND_ORDER_PARAMETERS
        self._template_bond_order(e1,e2,double_bond = double_bond, triple_bond = triple_bond, bounds = bounds)
        self._template_bond_energy_attractive(e1,e2,double_bond = double_bond, triple_bond = triple_bond, bounds = bounds)
        self._template_bond_energy_vdW(e1,e2,bounds = bounds)
        return

    def make_template_threebody(self, e1, e2, e3, bounds = 0.1):
        """
        Function to generate decision variables for all three-body terms for a given triplet of elements

        :param e1: Chemical symbol for element 1
        :type e1: str

        :param e2: Chemical symbol for element 2
        :type e2: str

        :param e3: Chemical symbol for element 3
        :type e3: str

        :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        :type bounds: float
        """
        self._template_threebody_energy(e1, e2, e3, bounds = bounds)
        return

    def make_template_fourbody(self, e1, e2, e3, e4, bounds = 0.1):
        """
        Function to generate decision variables for all four-body terms for a given quartet of elements

        :param e1: Chemical symbol for element 1
        :type e1: str

        :param e2: Chemical symbol for element 2
        :type e2: str

        :param e3: Chemical symbol for element 3
        :type e3: str

        :param e4: Chemical symbol for element 4
        :type e4: str

        :param bounds: Maximum deviation allowed for each decision variable from its current value in the forcefield
        :type bounds: float
        """
        self._template_fourbody_energy(e1, e2, e3, e4, bounds = bounds)
        return


    def write_formatted_forcefields(self):
        """
        Function to write-out the current forcefield with correct ReaxFF formatting

        :param outfilename: File to which formatted forcefield to be written to
        :type outfilename: str
        """
        string = self.header

        # Write general parameters
        for lineno, line in enumerate(self.general):
            if lineno == 0:
                string += ' %2d       %s\n' %(int(line[0]), ' '.join(line[1:]))
            else:
                string += '%10.4f %s\n' %(float(line[0]), ' '.join(line[1:]))

        # One-body term
        for lineno, line in enumerate(self.onebody[:4]):
            if lineno == 0:
                string += '%3d    %s\n' %(int(line[0]), ' '.join(line[1:]))
            else:
                string += '            %s\n' %(' '.join(line))

        for lineno, line in enumerate(self.onebody[4:]):
            if lineno % 4 == 0:
                string += ' %-2s' % line[0] + ''.join(['%9.4f' % float(val) for val in line[1:]]) + '\n'
            else:
                string += '   ' + ''.join(['%9.4f' % float(val) for val in line]) + '\n'

        # Two-body terms
        for lineno, line in enumerate(self.twobody[:2]):
            if lineno == 0:
                string += '%3d      %s\n' %(int(line[0]), ' '.join(line[1:]))
            else:
                string += '            %s\n' %(' '.join(line))

        for lineno, line in enumerate(self.twobody[2:]):
            if lineno % 2 == 0:
                string += '%3d' % int(line[0]) + '%3d' % int(line[1]) + ''.join(['%9.4f' % float(val) for val in line[2:]]) + '\n'
            else:
                string += '      ' + ''.join(['%9.4f' % float(val) for val in line]) + '\n'

        # Off-diagonal
        for lineno, line in enumerate(self.offdiagonal):
            if lineno == 0:
                string += '%3d    ' % int(line[0]) + ' '.join(line[1:]) + '\n'
            else:
                string += '%3d' % int(line[0]) + '%3d' % int(line[1]) + ''.join(['%9.4f' % float(val) for val in line[2:]]) + '\n'

        # Threebody
        for lineno, line in enumerate(self.threebody):
            if lineno == 0:
                string += '%3d    ' % int(line[0]) + ' '.join(line[1:]) + '\n'
            else:
                string += '%3d' % int(line[0]) + '%3d' % int(line[1]) + '%3d' % int(line[2]) + ''.join(['%9.4f' % float(val) for val in line[3:]]) + '\n'

        # Fourbody
        for lineno, line in enumerate(self.fourbody):
            if lineno == 0:
                string += '%3d    ' % int(line[0]) + ' '.join(line[1:]) + '\n'
            else:
                string += '%3d' % int(line[0]) + '%3d' % int(line[1]) + '%3d' % int(line[2]) + '%3d' % int(line[3]) + ''.join(['%9.4f' % float(val) for val in line[4:]])+ '\n'


        # Hbond
        for lineno, line in enumerate(self.hbond):
            if lineno == 0:
                string += '%3d    ' % int(line[0]) + ' '.join(line[1:]) + '\n'
            else:
                string += '%3d' % int(line[0]) + '%3d' % int(line[1]) + '%3d' % int(line[2]) + ''.join(['%9.4f' % float(val) for val in line[3:]]) + '\n'

        return string



    def write_gulp_library(self, outfilename = None):
        """
        Function to write-out the forcefield in the GULP library format

        :param outfilename: File to which the GULP ReaxFF forcefield library
        :type outfilename: str
        """
        # HEADER
        string = ''
        string += '#\n'
        string += '#  ReaxFF force field\n'
        string += '#\n'
        string += '#  Original paper:\n'
        string += '#\n'
        string += '#  A.C.T. van Duin, S. Dasgupta, F. Lorant and W.A. Goddard III,\n'
        string += '#  J. Phys. Chem. A, 105, 9396-9409 (2001)\n'
        string += '#\n'
        string += '#\n'

        # CUTOFFS
        string += '#  Cutoffs for VDW & Coulomb terms\n'
        string += '#\n'
        string += 'reaxFFvdwcutoff %12.4f\n' % float(self.general[13][0])
        string += 'reaxFFqcutoff   %12.4f\n' % float(self.general[13][0])
        string += '#\n'

        #BOND ORDER THRESHOLD
        string += '#  Bond order threshold - check anglemin as this is cutof2 given in control file\n'
        string += '#\n'
        string += 'reaxFFtol       %12.10f 0.001\n' % (float(self.general[30][0])*0.01)
        string += '#\n'

        #SPECIES INDEPENDENT PARAMETERS
        string += '#  Species independent parameters \n'
        string += '#\n'
        string += 'reaxff0_bond     %12.6f %12.6f\n' %(float(self.general[1][0]), float(self.general[2][0]))
        string += 'reaxff0_over     %12.6f %12.6f %12.6f %12.6f %12.6f\n' %(float(self.general[33][0]), float(self.general[32][0]), float(self.general[7][0]), float(self.general[9][0]), float(self.general[10][0]))
        string += 'reaxff0_valence  %12.6f %12.6f %12.6f %12.6f\n' %(float(self.general[15][0]), float(self.general[34][0]), float(self.general[17][0]), float(self.general[18][0]))
        string += 'reaxff0_penalty  %12.6f %12.6f %12.6f\n' %(float(self.general[20][0]), float(self.general[21][0]), float(self.general[22][0]))
        string += 'reaxff0_torsion  %12.6f %12.6f %12.6f %12.6f\n' %(float(self.general[24][0]), float(self.general[25][0]), float(self.general[26][0]), float(self.general[28][0]))
        string += 'reaxff0_vdw      %12.6f\n' % float(self.general[29][0])
        string += 'reaxff0_lonepair %12.6f\n' % float(self.general[16][0])
        string += '#\n'

        #SPECIES PARAMETERS - RADII
        string += '#  Species parameters \n'
        string += '#\n'
        string += 'reaxff1_radii\n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f %8.4f %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno][1]), float(self.onebody[lineno][7]), float(self.onebody[lineno+2][0]))

        #SPECIES PARAMETERS - VALENCE
        string += 'reaxff1_valence\n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f %8.4f %8.4f %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno][2]), float(self.onebody[lineno+3][3]), float(self.onebody[lineno][8]), float(self.onebody[lineno+1][2]))

        #SPECIES PARAMETERS - OVER
        string += 'reaxff1_over\n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f %8.4f %8.4f %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno+2][4]), float(self.onebody[lineno+2][3]), float(self.onebody[lineno+2][5]), float(self.onebody[lineno+3][0]))

        #SPECIES PARAMETERS - UNDER KCAL
        string += 'reaxff1_under kcal\n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno+1][3]))

        #SPECIES PARAMETERS - LONEPAIR KCAL
        string += 'reaxff1_lonepair kcal\n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f %8.4f\n' % (self.onebody[lineno][0], 0.5*(float(self.onebody[lineno][8]) - float(self.onebody[lineno][2])), float(self.onebody[lineno+2][1]))

        #SPECIES PARAMETERS - ANGLE
        string += 'reaxff1_angle\n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno+3][1]), float(self.onebody[lineno+3][4]))

        #SPECIES PARAMETERS - ANGLE
        string += 'reaxff1_morse kcal\n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f %8.4f %8.4f %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno+1][0]), float(self.onebody[lineno][5]), float(self.onebody[lineno][4]), float(self.onebody[lineno+1][1]))

        #ELEMENT PARAMETERS
        string += '#\n'
        string += '#  Element parameters \n'
        string += '#\n'

        #ELEMENT PARAMETERS - CHI
        string += 'reaxff_chi  \n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno+1][5]))

        #ELEMENT PARAMETERS - MU
        string += 'reaxff_mu   \n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno+1][6]))

        #ELEMENT PARAMETERS - MU
        string += 'reaxff_gamma  \n'
        for lineno in list(range(4,len(self.onebody),4)):
            string += '%-2s core %8.4f\n' % (self.onebody[lineno][0], float(self.onebody[lineno][6]))

        #BOND PARAMETERS
        string += '#\n'
        string += '#  Bond parameters \n'
        string += '#\n'

        #BOND PARAMETERS - BO OVER BO13
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        first = True
        for lineno in list(range(2,len(self.twobody),2)):
            if (float(self.twobody[lineno][7]) > 0.001 and float(self.twobody[lineno+1][6]) > 0.001):
                if first:
                    string += 'reaxff2_bo over bo13\n'
                    first = False
                n1, n2 = int(self.twobody[lineno][0]), int(self.twobody[lineno][1])
                bo3 = 0.0 if (np.absolute(float(self.twobody[lineno+1][1])-1.0) < 1.0e-12) else float(self.twobody[lineno+1][1])
                string += '%-2s core %-2s core %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], float(self.twobody[lineno+1][4]), float(self.twobody[lineno+1][5]), bo3, float(self.twobody[lineno+1][2]), float(self.twobody[lineno][6]), float(self.twobody[lineno][8]))

        #BOND PARAMETERS - BO UNDER
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        first = True
        for lineno in list(range(2,len(self.twobody),2)):
            if (float(self.twobody[lineno][7]) > 0.001 and float(self.twobody[lineno+1][6]) <= 0.001):
                if first:
                    string += 'reaxff2_bo bo13\n'
                    first = False
                n1, n2 = int(self.twobody[lineno][0]), int(self.twobody[lineno][1])
                bo3 = 0.0 if math.isclose(1.0, float(self.twobody[lineno+1][1])) else float(self.twobody[lineno+1][1])
                string += '%-2s core %-2s core %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], float(self.twobody[lineno+1][4]), float(self.twobody[lineno+1][5]), bo3, float(self.twobody[lineno+1][2]), float(self.twobody[lineno][6]), float(self.twobody[lineno][8]))

        #BOND PARAMETERS - BO OVER
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        first = True
        for lineno in list(range(2,len(self.twobody),2)):
            if (float(self.twobody[lineno][7]) <= 0.001 and float(self.twobody[lineno+1][6]) > 0.001):
                if first:
                    string += 'reaxff2_bo over\n'
                    first = False
                n1, n2 = int(self.twobody[lineno][0]), int(self.twobody[lineno][1])
                bo3 = 0.0 if math.isclose(1.0, float(self.twobody[lineno+1][1])) else float(self.twobody[lineno+1][1])
                string += '%-2s core %-2s core %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], float(self.twobody[lineno+1][4]), float(self.twobody[lineno+1][5]), bo3, float(self.twobody[lineno+1][2]), float(self.twobody[lineno][6]), float(self.twobody[lineno][8]))

        #BOND PARAMETERS - BO
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        first = True
        for lineno in list(range(2,len(self.twobody),2)):
            if (float(self.twobody[lineno][7]) <= 0.001 and float(self.twobody[lineno+1][6]) <= 0.001):
                if first:
                    string += 'reaxff2_bo \n'
                    first = False
                n1, n2 = int(self.twobody[lineno][0]), int(self.twobody[lineno][1])
                bo3 = 0.0 if math.isclose(1.0, float(self.twobody[lineno+1][1])) else float(self.twobody[lineno+1][1])
                string += '%-2s core %-2s core %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], float(self.twobody[lineno+1][4]), float(self.twobody[lineno+1][5]), bo3, float(self.twobody[lineno+1][2]), float(self.twobody[lineno][6]), float(self.twobody[lineno][8]))


        #BOND PARAMETERS - BOND KCAL
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        string += 'reaxff2_bond kcal \n'
        for lineno in list(range(2,len(self.twobody),2)):
            n1, n2 = int(self.twobody[lineno][0]), int(self.twobody[lineno][1])
            string += '%-2s core %-2s core %8.4f %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], float(self.twobody[lineno][2]), float(self.twobody[lineno][3]), float(self.twobody[lineno][4]), float(self.twobody[lineno][5]), float(self.twobody[lineno+1][0]))

        #BOND PARAMETERS - BOND OVER
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        string += 'reaxff2_over \n'
        for lineno in list(range(2,len(self.twobody),2)):
            n1, n2 = int(self.twobody[lineno][0]), int(self.twobody[lineno][1])
            string += '%-2s core %-2s core %8.4f \n' % (element_number[n1], element_number[n2], float(self.twobody[lineno][9]))

        #BOND PARAMETERS - BOND PEN KCAL
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        first = True
        for lineno in list(range(2,len(self.twobody),2)):
            if float(self.twobody[lineno+1][7]) > 0.0:
                if first:
                    string += 'reaxff2_pen kcal\n'
                    first = False
                n1, n2 = int(self.twobody[lineno][0]), int(self.twobody[lineno][1])
                string += '%-2s core %-2s core %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], float(self.twobody[lineno+1][7]), float(self.general[14][0]), 1.0)


        #BOND PARAMETERS - BOND MORSE KCAL
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        string += 'reaxff2_morse kcal\n'
        for line in self.offdiagonal[1:]:
            n1, n2 = int(line[0]), int(line[1])
            string += '%-2s core %-2s core %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], float(line[2]), float(line[4]), float(line[3]), float(line[5]), float(line[6]), float(line[7]))


        #ANGLE PARAMETERS
        string += '#\n'
        string += '#  Angle parameters \n'
        string += '#\n'


        #ANGLE PARAMETERS - ANGLE KCAL
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        string += 'reaxff3_angle kcal\n'
        for line in self.threebody[1:]:
            n2, n1, n3 = int(line[0]), int(line[1]), int(line[2])
            if float(line[4]) > 0.0:
                string += '%-2s core %-2s core %-2s core %8.4f %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], element_number[n3], float(line[3]), float(line[4]), float(line[5]), float(line[9]), float(line[7]))

        #ANGLE PARAMETERS - PENALTY KCAL
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        string += 'reaxff3_penalty kcal \n'
        for line in self.threebody[1:]:
            n2, n1, n3 = int(line[0]), int(line[1]), int(line[2])
            string += '%-2s core %-2s core %-2s core %8.4f\n' % (element_number[n1], element_number[n2], element_number[n3], float(line[8]))

        #ANGLE PARAMETERS - CONJUGATION KCAL
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        string += 'reaxff3_conjugation kcal \n'
        for line in self.threebody[1:]:
            if np.absolute(float(line[6])) > 1.0e-4:
                n2, n1, n3 = int(line[0]), int(line[1]), int(line[2])
                string += '%-2s core %-2s core %-2s core %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], element_number[n3], float(line[6]), float(self.general[3][0]), float(self.general[39][0]), float(self.general[31][0]))


        #HBOND PARAMETERS
        string += '#\n'
        string += '#  Hydrogen bond parameters \n'
        string += '#\n'

        #HBOND PARAMETERS - CONJUGATION KCAL
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        string += 'reaxff3_hbond kcal \n'
        for line in self.hbond[1:]:
            n2, n1, n3 = int(line[0]), int(line[1]), int(line[2])
            string += '%-2s core %-2s core %-2s core %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], element_number[n3], float(line[3]), float(line[4]), float(line[5]), float(line[6]))


        #TORSION PARAMETERS
        string += '#\n'
        string += '#  Torsion parameters \n'
        string += '#\n'

        #HBOND PARAMETERS - CONJUGATION KCAL
        element_number = ['X']
        for line in self.onebody[4::4]:
            element_number.append(line[0])
        string += 'reaxff4_torsion kcal \n'
        for line in self.fourbody[1:]:
            n1, n2, n3, n4 = int(line[0]), int(line[1]), int(line[2]), int(line[3])
            string += '%-2s core %-2s core %-2s core %-2s core %8.4f %8.4f %8.4f %8.4f %8.4f\n' % (element_number[n1], element_number[n2], element_number[n3], element_number[n4], float(line[4]), float(line[5]), float(line[6]), float(line[7]), float(line[8]))


        #GENERAL PARAMETERS - Thresholds and cutoffs
        string += '\n\n'
        string += 'reaxfftol {0:.12f}'.format(self.bo_threshold)
        string += '\n\n'

        if outfilename is not None:
            libfile = open(outfilename, 'w')
            libfile.write(string)
            libfile.close()
            return
        else:
            return string
