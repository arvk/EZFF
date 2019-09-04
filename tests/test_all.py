import ezff
import xtal
from ezff.interfaces import qchem
import pytest
import numpy as np

def test_qchem_read_energy_singlepoint():
    energy = qchem.read_energy('qchem_output_files/optCHOSx.out')
    is_correct_type = isinstance(energy, np.ndarray)
    is_correct_size = (len(energy) == 1)
    is_correct_value = (np.linalg.norm(energy) == pytest.approx(14044.3073))
    assert (is_correct_type and is_correct_size and is_correct_value)


def test_qchem_read_structure_singlepoint():
    structure = qchem.read_structure('qchem_output_files/optCHOSx.out')
    has_correct_snapsize = (len(structure.snaplist) == 1)
    has_correct_atomsize = (len(structure.snaplist[0].atomlist) == 10)
    has_correct_data_1 = (np.linalg.norm(structure.snaplist[0].atomlist[0].cart - structure.snaplist[0].atomlist[1].cart) == pytest.approx(1.6389119))
    assert (has_correct_snapsize and has_correct_atomsize and has_correct_data_1)
