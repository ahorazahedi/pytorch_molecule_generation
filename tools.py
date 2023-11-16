"""
Gets the atom types present in a set of molecules.

To use script, run:
python atom_types.py --smi path/to/file.smi
"""
import argparse
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier
from tqdm import tqdm

def load_molecules(path : str) -> rdkit.Chem.rdmolfiles.SmilesMolSupplier:

    with open(path) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    molecule_set = SmilesMolSupplier(path, sanitize=True, nameColumn=-1, titleLine=has_header)

    return molecule_set

def get_atom_types(smi_file : str) -> list:
    molecules = load_molecules(path=smi_file)

    atom_types = list()
    for mol in tqdm(molecules , desc="Calculation Atom Types"):
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetAtomicNum())

    set_of_atom_types = set(atom_types)
    atom_types_sorted = list(set_of_atom_types)
    atom_types_sorted.sort()

    return [rdkit.Chem.Atom(atom).GetSymbol() for atom in atom_types_sorted]


def get_formal_charges(smi_file : str) -> list:
   
    molecules = load_molecules(path=smi_file)

    formal_charges = list()
    for mol in tqdm(molecules , desc="Calculation Formal Charge"):
        for atom in mol.GetAtoms():
            formal_charges.append(atom.GetFormalCharge())

    set_of_formal_charges = set(formal_charges)
    formal_charges_sorted = list(set_of_formal_charges)
    formal_charges_sorted.sort()

    return formal_charges_sorted



def get_max_n_atoms(smi_file : str) -> int:
  
    molecules = load_molecules(path=smi_file)

    max_n_atoms = 0
    for mol in tqdm(molecules , desc="Calculation Max number of Atoms"):
        n_atoms = mol.GetNumAtoms()

        if n_atoms > max_n_atoms:
            max_n_atoms = n_atoms

    return max_n_atoms

