import os
import numpy as np
import rdkit
import h5py
from tqdm import tqdm

import itertools
import torch
import random
from rdkit.Chem.rdmolfiles import MolToSmiles


# load general packages and functions
from collections import namedtuple
import itertools
import random
from copy import deepcopy
from typing import Union, Tuple
import numpy as np
import rdkit



# load GraphINVENT-specific functions
# from Analyzer import Analyzer
# from Parameters import Parameters as constants
# from Parameters import Parameters as C
# # import load_molecules as load
# from Utils import molecules as load_molecules

# import Utils as util

# from Graphs.utils import get_feature_vector_indices , one_of_k_encoding

class MolecularGraph:
    """
    Parent class for all molecular graphs.

    This class is then inherited by three subclasses:
      `PreprocessingGraph`, which is used when preprocessing training data, and
      `TrainingGraph`, which is used when training structures.
      `GenerationGraph`, which is used when generating structures.

    The reason for the two classes is that `np.ndarray`s are needed to save
    test/train/valid sets to HDF file format when preprocessing, but for
    training it is more efficient to use `torch.Tensor`s since these can be
    easily used on the GPU for training/generation.
    """
    def __init__(self, constants : namedtuple,
                 molecule : rdkit.Chem.Mol ,
                 node_features : Union[np.ndarray, torch.Tensor],
                 edge_features : Union[np.ndarray, torch.Tensor],
                 atom_feature_vector : torch.Tensor) -> None:
        """
        Args:
        ----
            constants (namedtuple)             : Contains job parameters as
                                                 well as global constants.
            molecule (rdkit.Chem.Mol)          : Input used for creating
                                                 `PreprocessingGraph`.
            atom_feature_vector (torch.Tensor) : Input used for creating
                                                 `TrainingGraph`.
        """
        self.constants = constants

        # placeholders (these are set in the respective sub-classes)
        self.molecule      = None
        self.node_features = None
        self.edge_features = None
        self.n_nodes       = None

    def get_graph_state(self) -> None:
        """
        This function is implemented in each subclass, since preprocessing
        graphs use `np.ndarray`s and training/generation graphs use
        `torch.Tensor`s.
        """
        raise NotImplementedError

    def get_n_edges(self) -> int:
        """
        Gets the number of edges in the `MolecularGraph`.
        """
        # divide by 2 to avoid double-counting edges
        n_edges = self.edge_features.sum() // 2
        return n_edges

    def get_molecule(self) -> rdkit.Chem.Mol:
        """
        Gets the molecule representation of the current `MolecularGraph`.
        """
        if self.molecule is False:
            pass
        else:
            self.molecule = self.graph_to_mol()

        return self.molecule

    def get_smiles(self) -> str:
        """
        Gets the SMILES representation of the current `MolecularGraph`.
        """
        try:
            smiles = MolToSmiles(mol=self.molecule, kekuleSmiles=False)
        except:
            # if molecule is invalid, set SMILES to `None`
            smiles = None
        return smiles

    def graph_to_mol(self) -> rdkit.Chem.Mol:
        """
        Generates the `rdkit.Chem.Mol` object corresponding to the graph.

        The function uses for a given graph:
          * `n_nodes`       : number of nodes in graph
          * `node_features` : node feature matrix, a |V|x|NF| matrix
          * `edge_features` : edge feature tensor, a |V|x|V|x|B| tensor

        Above, V is the set of all nodes in a graph, NF is the set of node
        features, and B is the set of available bond types.
        """
        # create empty editable `rdkit.Chem.Mol` object
        molecule    = rdkit.Chem.RWMol()

        # add atoms to `rdkit.Chem.Mol` and keep track of idx
        node_to_idx = {}

        for v in range(0, self.n_nodes):
            atom_to_add    = self.features_to_atom(node_idx=v)
            molecule_idx   = molecule.AddAtom(atom_to_add)
            node_to_idx[v] = molecule_idx

        # add bonds between adjacent atoms
        for bond_type in range(self.constants.n_edge_features):
            # `self.edge_features[:, :, bond_type]` is an adjacency matrix
            #  for that specific `bond_type`
            for bond_idx1, row in enumerate(
                self.edge_features[:self.n_nodes, :self.n_nodes, bond_type]
                ):
                # traverse only half adjacency matrix to not duplicate bonds
                for bond_idx2 in range(bond_idx1):
                    bond = row[bond_idx2]
                    if bond:  # if `bond_idx1` and `bond_idx2` are bonded
                        try:  # try adding the bond to `rdkit.Chem.Mol` object
                            molecule.AddBond(
                                node_to_idx[bond_idx1],
                                node_to_idx[bond_idx2],
                                self.constants.int_to_bondtype[bond_type]
                            )
                        except (TypeError, RuntimeError, AttributeError):
                            # errors occur if the above `AddBond()` action tries
                            # to add multiple bonds to a node pair (should not
                            # happen, but kept here as a safety)
                            raise ValueError("MolecularGraphError: Multiple "
                                             "edges connecting a single pair "
                                             "of nodes in graph.")

        try:  # convert from `rdkit.Chem.RWMol` to Mol object
            molecule.GetMol()
        except AttributeError:  # raised if molecules is `None`
            pass

        # if `ignore_H` flag is used, "sanitize" the structure to correct
        # the number of implicit hydrogens (otherwise, they will all stay at 0)
        if self.constants.ignore_H and molecule:
            try:
                rdkit.Chem.SanitizeMol(molecule)
            except ValueError:
                # raised if `molecule` is False, None, or too ugly to sanitize
                pass

        return molecule

    def features_to_atom(self, node_idx : int) -> rdkit.Chem.Atom:
        """
        Converts the atomic feature vector corresponding to the atom indicated
        by input `node_idx` into an `rdkit.Atom` object.

        The atomic feature vector describes a unique node on a graph using
        concatenated one-hot encoded vectors for the features of interest (e.g.
        atom type, formal charge), and is a row of `self.node_features`. Note
        that if `ignore_H` flag is used, will assign a placeholder of 0 to the
        number of implicit hydrogens in the atom (to be corrected for later via
        kekulization).

        Args:
        ----
            node_idx (int) : Index for a node feature vector (i.e. a row of
                             `self.node_features`).

        Returns:
        -------
            new_atom (rdkit.Atom) : Atom object.
        """
        # determine the nonzero indices of the feature vector
        feature_vector = self.node_features[node_idx]
        try:  # if `feature_vector` is a `torch.Tensor`
            nonzero_idc = torch.nonzero(feature_vector)
        except TypeError:  # if `feature_vector` is a `numpy.ndarray`
            nonzero_idc = np.nonzero(feature_vector)[0]

        # determine atom symbol
        atom_idx  = nonzero_idc[0]
        atom_type = self.constants.atom_types[atom_idx]
        new_atom  = rdkit.Chem.Atom(atom_type)

        # determine formal charge
        fc_idx        = nonzero_idc[1] - self.constants.n_atom_types
        formal_charge = self.constants.formal_charge[fc_idx]

        new_atom.SetFormalCharge(formal_charge)

        # determine number of implicit Hs
        if not self.constants.use_explicit_H and not self.constants.ignore_H:
            total_num_h_idx = (
                nonzero_idc[2]
                - self.constants.n_atom_types
                - self.constants.n_formal_charge
            )
            total_num_h = self.constants.imp_H[total_num_h_idx]

            new_atom.SetUnsignedProp("_TotalNumHs", total_num_h)

        elif self.constants.ignore_H:
            # Hs will be set with structure is sanitized later
            pass

        # determine chirality
        if self.constants.use_chirality:
            cip_code_idx = (
                    nonzero_idc[-1]
                    - self.constants.n_atom_types
                    - self.constants.n_formal_charge
                    - bool(not self.constants.use_explicit_H and
                           not self.constants.ignore_H)
                    * self.constants.n_imp_H
            )
            cip_code = self.constants.chirality[cip_code_idx]
            new_atom.SetProp("_CIPCode", cip_code)

        return new_atom

    def mol_to_graph(self, molecule : rdkit.Chem.Mol) -> None:
        """
        Generates the graph representation (`self.node_features` and
        `self.edge_features`) when creating a new `PreprocessingGraph`.
        """
        n_atoms = self.n_nodes
        atoms   = map(molecule.GetAtomWithIdx, range(n_atoms))

        # build the node features matrix using a Numpy array
        node_features = np.array(list(map(self.atom_features, atoms)),
                                 dtype=np.int32)

        # build the edge features tensor using a Numpy array
        edge_features = np.zeros(
            [n_atoms, n_atoms, self.constants.n_edge_features],
            dtype=np.int32
        )
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = self.constants.bondtype_to_int[bond.GetBondType()]
            edge_features[i, j, bond_type] = 1
            edge_features[j, i, bond_type] = 1

        # define the number of nodes
        self.n_nodes = n_atoms

        self.node_features = node_features  # not padded!
        self.edge_features = edge_features  # not padded!

