import sys
sys.path.append('..')
import ezff
from ezff.interfaces import gulp
import pytest

# TEST error_structure_distortion
def test_structure_distortion_nobox():
    test_value = gulp.error_structure_distortion('gulp_output_files/out.mote2.nobox', relax_atoms=True, relax_cell=False)
    ground_truth = pytest.approx(5.171,0.01)
    assert (test_value == ground_truth)

def test_structure_distortion_nooptim():
    test_value = gulp.error_structure_distortion('gulp_output_files/out.mote2.nooptim', relax_atoms=False, relax_cell=False)
    ground_truth = 0.0
    assert (test_value == ground_truth)

def test_structure_distortion_optim():
    test_value = gulp.error_structure_distortion('gulp_output_files/out.mote2.optim', relax_atoms=True, relax_cell=False)
    ground_truth = pytest.approx(0.11,0.01)
    assert (test_value == ground_truth)

def test_structure_distortion_sheared():
    test_value = gulp.error_structure_distortion('gulp_output_files/out.mote2.sheared', relax_atoms=True, relax_cell=True)
    ground_truth = pytest.approx(26.48,0.01)
    assert (test_value == ground_truth)

def test_structure_distortion_lj():
    test_value = gulp.error_structure_distortion('gulp_output_files/out.lj.optim', relax_atoms=True, relax_cell=True)
    ground_truth = pytest.approx(0.03,0.5)
    assert (test_value == ground_truth)

def test_structure_distortion_reaxff_singlepoint():
    test_value = gulp.error_structure_distortion('gulp_output_files/out.reaxff.singlepoint', relax_atoms=False, relax_cell=False)
    ground_truth = 0.0
    assert (test_value == ground_truth)
