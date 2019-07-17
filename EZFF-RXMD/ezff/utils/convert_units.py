# Energy conversion
eV = {'eV':1.0, 'meV':1000.0, 'kcal/mol':23.0605, 'kJ/mol':96.4853, 'Ha':0.03674932248, 'Ry':0.07349864496} # From eV
meV = {key: eV[key]/eV['meV'] for key in eV.keys()} # From meV
kcal_mol = {key: eV[key]/eV['kcal/mol'] for key in eV.keys()} # From kcal/mol
kJ_mol = {key: eV[key]/eV['kJ/mol'] for key in eV.keys()} # From kJ/mol
Ha = {key: eV[key]/eV['Ha'] for key in eV.keys()} # From Ha
Ry = {key: eV[key]/eV['Ry'] for key in eV.keys()} # From Ry

energy = {'eV':eV, 'meV':meV, 'kcal/mol':kcal_mol, 'kJ/mol':kJ_mol, 'Ha':Ha, 'Ry':Ry}
#del eV meV kcal_mol kJ_mol Ha Ry



# Frequency conversion
THz = {'THz':1.0, 'cm-1':33.35641, 'meV':4.13567, 'nm':299792.27794, 'um':299.79228, 'eV':0.00414, 'fs':1000.0, 'ps':1.0} # From THz
cm_1 = {key: THz[key]/THz['cm-1'] for key in THz.keys()} # From cm-1
meV = {key: THz[key]/THz['meV'] for key in THz.keys()} # From meV
nm = {key: THz[key]/THz['nm'] for key in THz.keys()} # From nm
um = {key: THz[key]/THz['um'] for key in THz.keys()} # From um
eV = {key: THz[key]/THz['eV'] for key in THz.keys()} # From eV
fs = {key: THz[key]/THz['fs'] for key in THz.keys()} # From fs
ps = {key: THz[key]/THz['ps'] for key in THz.keys()} # From ps

frequency = {'THz':THz, 'cm-1':cm_1, 'meV':meV, 'nm':nm, 'um':um, 'eV':eV, 'fs':fs, 'ps':ps}
#del THz cm_1 meV nm um, eV fs ps
