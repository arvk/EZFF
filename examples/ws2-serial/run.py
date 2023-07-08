from functools import wraps
from math import sqrt
from random import seed

import numpy as np
from sklearn.metrics import mean_squared_error as f_err
from sklearn.metrics import mean_squared_error as mse

import ezff
from ezff.interfaces import gulp, vasp
from ezff.interfaces.gulp import job

gulp_path = r'/home/anton/gulp/gulp-6.1.1/Exe/gulp'

#Additional optional functions
def check_values(values):
    _ = []
    for i in values:
        i_ = i
        if str(i) == 'nan':
            i_ = 0
        if "*" in str(i):
            i_ = 0
        _.append(i_)
    return _

def get_abc(structure):
    _ = list(i if (i != float('nan') or i != float('infinity') ) else 0 for i in structure.abc)
    return _

def get_ang(structure):
    _ = list(structure.ang)
    return _

def get_coords(structure):
    _ = list([list(i.cart) for i in structure.snaplist[-1].atomlist])
    return _

def get_structure_info(structures):
    abc, ang, coord = [], [], []

    for structure in structures:
        structure.make_dircar_matrices()
        structure.box_to_abc()
        abc.append(get_abc(structure))
        ang.append(get_ang(structure))
        coord.append(get_coords(structure))

    return abc, ang, coord

def _calculate_rmsd(GT_freqs, MD_freqs):
    return sqrt(mse(GT_freqs, MD_freqs))

def get_atoms(structure):
    return np.array([i.cart for i in gt_H2_structure.snaplist[-1].atomlist])

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    if isinstance(S[0], np.ndarray):
        return flatten(list(S[0])) + flatten(list(S[1:]))
    return S[:1] + flatten(S[1:])


# DEFINE GROUND TRUTHS
gt_H2_structure = vasp.read_atomic_structure(r'/home/anton/Coding/EZFF2/examples/ws2-serial/ground_truths/WS2_2H_3R_1H_1T_AB_g60_3D_1.POSCAR')
gt_H2_energy = -22.328
gt_H2_c11 = 158.0 #GPa for C11 of H2-WS2
gt_H2_c33 = 60.0 #GPa for C11 of H2-WS2
gt_H2_c44 = 16.0 #GPa for C11 of H2-WS2
gt_H2_bulk_modulus = 44.00
gt_H2_freq= [0, 0, 0, 27.4, 27.4, 45, 306, 306, 306, 306] # 10
gt_H2_young_x = 160.0
gt_H2_young_y = 160.0
H2_sshift=1

gt_R3_structure = vasp.read_atomic_structure(r'/home/anton/Coding/EZFF2/examples/ws2-serial/ground_truths/WS2_2H_3R_1H_1T_AB_g60_3D_2.POSCAR')
gt_R3_energy = -33.486
R3_sshift = 1.5

gt_H1_structure = vasp.read_atomic_structure(r'/home/anton/Coding/EZFF2/examples/ws2-serial/ground_truths/WS2_2H_3R_1H_1T_AB_g60_3D_3.POSCAR')
gt_H1_energy = -10.930
gt_H1_freq= [0, 0, 0, 300, 300, 357, 357, 419, 444] # 9
H1_sshift = 0.5

gt_T1_structure = vasp.read_atomic_structure(r'/home/anton/Coding/EZFF2/examples/ws2-serial/ground_truths/WS2_2H_3R_1H_1T_AB_g60_3D_4.POSCAR')
gt_T1_energy = -10.027
T1_sshift = 0.5

abc, ang, coord = get_structure_info([gt_H2_structure, gt_R3_structure, gt_H1_structure, gt_T1_structure])

gt_abc_err = flatten(abc)

del gt_abc_err[8]

gt_ang_err = flatten(ang)
gt_atom_err = flatten(coord)

gt_mech_err = flatten([gt_H2_young_x, gt_H2_young_y, gt_H2_c11, gt_H2_c33, gt_H2_c44])
gt_energy_err = flatten([gt_H2_energy, gt_R3_energy, gt_H1_energy, gt_T1_energy])
gt_freq_err = flatten([gt_H2_freq, gt_H1_freq])

h2_c, r3_c, t1_c = abc[0][2], abc[1][2], abc[3][2]

def my_error_function(variable_values:dict):

    # Error lists:
    # 1) Вектора трансляции
    # 2) Углы решетки
    # 3) РМСД положения атомов
    # 4) Частоты колебаний
    # 5) Механические свойства (тензор упругости, модуль юнга, bulk модуль)
    # 6) Энергия
    md_abc_err, md_ang_err, md_atom_err, md_freq_err, md_mech_err, md_energy_err = [], [], [], [], [], []

    timer = 15

    # Get rank from pool

    myrank = 0

    # FOR THE RELAXED STRUCTURE 1
    path = str(myrank)+'/1'

    # H2
    H2_job = job(path=path)
    H2_job.structure = gt_H2_structure
    H2_job.forcefield = ezff.generate_forcefield(template, variable_values, FFtype = 'SW')
    H2_job.options['pbc'] = True
    H2_job.options['relax_atoms'] = True
    H2_job.options['relax_cell'] = True
    H2_job.options['sshift'] = H2_sshift
    H2_job.options['phonon_G'] = True
    # freqs
    # Submit job and read output
    H2_job.run(gulp_path, timeout=timer) # Что выдаёт в случае неудачи?

    md_H2_structure = H2_job.read_structure()
    md_H2_moduli = H2_job.read_elastic_moduli()
    md_H2_yx, md_H2_yy, md_H2_yz = H2_job.read_young_xyz()
    md_H2_freqs = H2_job.read_phonon_G(10)
    md_H2_bulk_modulus, _ = H2_job.read_bulk_shear()

    try:
        md_H2_energy = H2_job.read_energy()[-1]
    except:
        md_H2_energy = 250

    md_mech_err = [md_H2_yx,
                md_H2_yy,
                md_H2_moduli[0][0,0],
                md_H2_moduli[0][2,2],
                md_H2_moduli[0][3,3],]

    md_energy_err.append(md_H2_energy)
    md_freq_err.append(md_H2_freqs)

    H2_job.cleanup()

    path = str(myrank)+'/2'

    # R3
    R3_job = job(path=path)
    R3_job.structure = gt_R3_structure
    R3_job.forcefield = ezff.generate_forcefield(template, variable_values, FFtype = 'SW')
    R3_job.options['pbc'] = True
    R3_job.options['relax_atoms'] = True
    R3_job.options['relax_cell'] = True
    R3_job.options['sshift'] = R3_sshift
    R3_job.run(gulp_path, timeout=timer)

    md_R3_structure = R3_job.read_structure()

    try:
        md_R3_energy = R3_job.read_energy()[-1]
    except:
        md_R3_energy = 250

    md_energy_err.append(md_R3_energy)

    R3_job.cleanup()

    path = str(myrank)+'/3'

    # H1
    H1_job = job(path=path)
    H1_job.structure = gt_H1_structure
    H1_job.forcefield = ezff.generate_forcefield(template, variable_values, FFtype = 'SW')
    H1_job.options['pbc'] = True
    H1_job.options['relax_atoms'] = True
    H1_job.options['relax_cell'] = True
    H1_job.options['sshift'] = H1_sshift
    H1_job.options['phonon_G'] = True
    H1_job.run(gulp_path, timeout=timer)

    md_H1_structure = H1_job.read_structure()
    md_H1_freqs = H1_job.read_phonon_G(9)

    try:
        md_H1_energy = H1_job.read_energy()[-1]
    except:
        md_H1_energy = 250


    md_energy_err.append(md_H1_energy)
    md_freq_err.append(md_H1_freqs)

    H1_job.cleanup()

    path = str(myrank)+'/4'

    # T1
    T1_job = job(path=path)
    T1_job.structure = gt_T1_structure
    T1_job.forcefield = ezff.generate_forcefield(template, variable_values, FFtype = 'SW')
    T1_job.options['pbc'] = True
    T1_job.options['relax_atoms'] = True
    T1_job.options['relax_cell'] = True
    T1_job.options['sshift'] = T1_sshift
    T1_job.run(gulp_path, timeout=timer)

    md_T1_structure = T1_job.read_structure()

    try:
        md_T1_energy = T1_job.read_energy()[-1]
    except:
        md_T1_energy = 250


    md_energy_err.append(md_T1_energy)

    T1_job.cleanup()

    abc, ang, coord = get_structure_info([md_H2_structure, md_R3_structure, md_H1_structure, md_T1_structure])

    md_abc_err, md_ang_err, md_atom_err = check_values(flatten(abc)), check_values(flatten(ang)), check_values(flatten(coord))
    md_freq_err, md_mech_err, md_energy_err = flatten(md_freq_err), flatten(md_mech_err), flatten(md_energy_err)
    del md_abc_err[8]

    x1 = f_err(gt_abc_err, md_abc_err)
    x2 = f_err(gt_ang_err, md_ang_err)
    # x3 = f_err(gt_atom_err, md_atom_err)
    x4 = f_err(gt_freq_err, md_freq_err)
    x5 = f_err(gt_mech_err, md_mech_err)
    x6 = f_err(gt_energy_err, md_energy_err)

    mh2_c, mr3_c, mt1_c = abc[0][2], abc[1][2], abc[3][2]

    return x1, x2, x4, x5, x6

n_iter = 10
population= 10
alg_name = 'IBEA'
mutatuion_rate=0.01
seed_=1910

bound = 'variable_bounds_AV_lin'
bound_name = bound.split('.')[0]

bounds = ezff.read_variable_bounds(f'/home/anton/Coding/EZFF2/examples/ws2-serial/ground_truths/{bound}', verbose=False)
template = ezff.read_forcefield_template(r'/home/anton/Coding/EZFF2/examples/ws2-serial/ground_truths/template_AV_lin')

seed(seed_)
f_err_name = f_err.__name__

problem = ezff.OptProblem(num_errors = 5,
                          variable_bounds = bounds,
                          error_function = my_error_function,
                          template = template)

algorithm = ezff.Algorithm(problem,
                           alg_name,
                           population = population)

ezff.optimize(problem,
              algorithm,
              iterations = n_iter)
