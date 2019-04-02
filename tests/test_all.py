import sys
sys.path.append('..')
import ezff
from ezff.interfaces import qchem
import pytest
import numpy as np

def test_qchem_read_energy_singlepoint():
    energy = qchem.read_energy('qchem_output_files/optCHOSx.out')
    is_correct_type = isinstance(energy, np.ndarray)
    is_correct_size = (len(energy) == 1)
    is_correct_value = (np.linalg.norm(energy) == pytest.approx(14044.3073))
    assert (is_correct_type and is_correct_size and is_correct_value)
