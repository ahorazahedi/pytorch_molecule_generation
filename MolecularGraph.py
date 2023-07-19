import itertools
import numpy as np
import torch
import random
import rdkit
from rdkit.Chem.rdmolfiles import MolToSmiles
from parameters import Parameters as C


def get_feature_vector_indices():

    idc = [C.n_atom_types, C.n_formal_charge]

    # indices corresponding to implicit H's and chirality are optional (below)
    if not C.use_explicit_H and not C.ignore_H:
        idc.append(C.n_imp_H)

    if C.use_chirality:
        idc.append(C.n_chirality)

    return np.cumsum(idc).tolist()


def one_of_k_encoding(x, allowable_set):

    if x not in set(allowable_set):  # use set for speedup over list
        raise Exception(
            f"Input {x} not in allowable set {allowable_set}. "
            f"Add {x} to allowable set in `features.py` and run again."
        )

    one_hot_generator = (int(x == s) for s in allowable_set)

    return one_hot_generator


class MolecularGraph:

    def __init__(self, constants, molecule, node_features, edge_features, atom_feature_vector):

        if constants:
            self.C = constants
            self.n_edge_features = constants.n_edge_features
        else:
            self.n_edge_features = None

        # placeholders
        self.molecule = None
        self.node_features = None
        self.edge_features = None
        self.n_nodes = None
        self.node_features = None
        self.edge_features = None

    def get_graph_state(self):

        raise NotImplementedError

    def get_n_edges(self):

        n_edges = self.edge_features.sum() / 2.0

        return int(n_edges)  # return an int

    def get_molecule(self):

        if self.molecule is False:
            pass
        else:
            self.molecule = self.graph_to_mol()

        return self.molecule

    def get_smiles(self):

        try:
            smiles = MolToSmiles(mol=self.molecule,
                                 kekuleSmiles=False)
        except:  # if molecule is invalid, set SMILES to `None`
            smiles = None

        return smiles

    def graph_to_mol(self):

        # create empty editable `rdkit.Chem.Mol` object
        molecule = rdkit.Chem.RWMol()

        # add atoms to `rdkit.Chem.Mol` and keep track of idx
        node_to_idx = {}

        for v in range(0, self.n_nodes):
            atom_to_add = self.features_to_atom(node_idx=v)
            molecule_idx = molecule.AddAtom(atom_to_add)
            node_to_idx[v] = molecule_idx

        # add bonds between adjacent atoms
        for bond_type in range(self.n_edge_features):
            #  for that specific `bond_type`
            for vi, row in enumerate(self.edge_features[:self.n_nodes, :self.n_nodes, bond_type]):
                # traverse only half adjacency matrix to not duplicate bonds
                for vj in range(vi):
                    bond = row[vj]
                    if bond:  # if `vi` and `vj` are bonded
                        try:  # try adding the bond to `rdkit.Chem.Mol` object
                            molecule.AddBond(node_to_idx[vi],
                                             node_to_idx[vj],
                                             self.C.int_to_bondtype[bond_type])
                        except (TypeError, RuntimeError, AttributeError):
                            raise ValueError("MolecularGraphError: Multiple"
                                             " edges connecting a single pair"
                                             " of nodes in graph.")

        try:
            molecule.GetMol()
        except AttributeError:  # raised if molecules is `None`
            pass

        if self.C.ignore_H and molecule:
            try:
                rdkit.Chem.SanitizeMol(molecule)
            except ValueError:
                pass

        return molecule

    def features_to_atom(self, node_idx):

        feature_vector = self.node_features[node_idx]
        try:  # if `feature_vector` is a `torch.Tensor`
            nonzero_idc = torch.nonzero(feature_vector)
        except TypeError:  # if `feature_vector` is a `numpy.ndarray`
            nonzero_idc = np.nonzero(feature_vector)[0]

        atom_idx = nonzero_idc[0]
        atom_type = self.C.atom_types[atom_idx]

        new_atom = rdkit.Chem.Atom(atom_type)

        fc_idx = nonzero_idc[1] - self.C.n_atom_types
        formal_charge = self.C.formal_charge[fc_idx]

        new_atom.SetFormalCharge(formal_charge)

        # determine number of implicit Hs
        if not self.C.use_explicit_H and not self.C.ignore_H:
            total_num_h_idx = (
                nonzero_idc[2]
                - self.C.n_atom_types
                - self.C.n_formal_charge
            )
            total_num_h = self.C.imp_H[total_num_h_idx]

            new_atom.SetUnsignedProp("_TotalNumHs", total_num_h)

        elif self.C.ignore_H:
            pass

        if self.C.use_chirality:
            cip_code_idx = (
                    nonzero_idc[-1] - self.C.n_atom_types - self.C.n_formal_charge
                    - bool(not self.C.use_explicit_H and not self.C.ignore_H)
                    * self.C.n_imp_H
            )
            cip_code = self.C.chirality[cip_code_idx]
            new_atom.SetProp("_CIPCode", cip_code)

        return new_atom

    def mol_to_graph(self, molecule):

        n_atoms = self.n_nodes
        atoms = map(molecule.GetAtomWithIdx, range(n_atoms))

        node_features = np.array(list(map(self.atom_features, atoms)), dtype=np.int32)

        edge_features = np.zeros([n_atoms, n_atoms, self.n_edge_features], dtype=np.int32)
        for bond in molecule.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = self.C.bondtype_to_int[bond.GetBondType()]
            edge_features[i, j, bond_type] = 1
            edge_features[j, i, bond_type] = 1

        self.n_nodes = n_atoms

        self.node_features = node_features  # not padded!
        self.edge_features = edge_features  # not padded!


class PreprocessingGraph(MolecularGraph):
    
    def __init__(self, constants, molecule):
        super(PreprocessingGraph, self).__init__(constants, molecule=False,
                                                 node_features=False,
                                                 edge_features=False,
                                                 atom_feature_vector=False)

        self.node_ordering = None 

        if self.C.use_explicit_H and not self.C.ignore_H:
            molecule = rdkit.Chem.AddHs(molecule)

        self.n_nodes = molecule.GetNumAtoms()

        self.mol_to_graph(molecule=molecule)

        self.node_remap(molecule=molecule)

        self.pad_graph_representation()

    def atom_features(self, atom):
        feature_vector_generator = itertools.chain(
            one_of_k_encoding(atom.GetSymbol(), self.C.atom_types),
            one_of_k_encoding(atom.GetFormalCharge(), self.C.formal_charge)
        )
        if not self.C.use_explicit_H and not self.C.ignore_H:
            feature_vector_generator = itertools.chain(
                feature_vector_generator,
                one_of_k_encoding(atom.GetTotalNumHs(), self.C.imp_H)
            )
        if self.C.use_chirality:
            try:
                chiral_state = atom.GetProp("_CIPCode")
            except KeyError:
                chiral_state = self.C.chirality[0]  # "None"

            feature_vector_generator = itertools.chain(
                feature_vector_generator,
                one_of_k_encoding(chiral_state, self.C.chirality)
            )

        feature_vector = np.fromiter(feature_vector_generator, int)

        return feature_vector

    def breadth_first_search(self, node_ranking, node_init=0):
       
        nodes_visited = [node_init]
        last_nodes_visited = [node_init]

        # loop until all nodes have been visited
        while len(nodes_visited) < self.n_nodes:
            neighboring_nodes = []

            for node in last_nodes_visited:
                neighbor_nodes = []
                for bond_type in range(self.n_edge_features):
                    neighbor_nodes.extend(list(
                        np.nonzero(self.edge_features[node, :, bond_type])[0]
                    ))
                new_neighbor_nodes = list(
                    set(neighbor_nodes) - (set(neighbor_nodes) & set(nodes_visited))
                )
                node_importance = [node_ranking[neighbor_node] for
                                   neighbor_node in new_neighbor_nodes]

                # check all neighboring nodes and sort in order of importance
                while sum(node_importance) != -len(node_importance):
                    next_node = node_importance.index(max(node_importance))
                    neighboring_nodes.append(new_neighbor_nodes[next_node])
                    node_importance[next_node] = -1

            # append the new, sorted neighboring nodes to list of visited nodes
            nodes_visited.extend(set(neighboring_nodes))

            # update the list of most recently visited nodes
            last_nodes_visited = set(neighboring_nodes)

        return nodes_visited

    def node_remap(self, molecule):
      
        if not self.C.use_canon:
            # get a random node ranking
            atom_ranking = list(range(self.n_nodes))
            random.shuffle(atom_ranking)
        else:
            # get RDKit canonical ranking
            atom_ranking = list(rdkit.Chem.CanonicalRankAtoms(molecule, breakTies=True))

        # using a random node as a starting point, get a new node ranking that
        # does not leave isolated fragments in graph traversal
        self.node_ordering = self.breadth_first_search(
            node_ranking=atom_ranking,
            node_init=atom_ranking[0]
        )

        # reorder all nodes according to new node ranking
        self.reorder_nodes()

    def get_decoding_APD(self):
        # **note: "idx" = "idx", "idc" == "indices"
        last_node_idx = self.n_nodes - 1  # zero-indexing

        # determine the indices of the atom descriptors # (i.e. atom type)
        fv_nonzero_idc = self.get_nonzero_feature_indices(node_idx=last_node_idx)

        # initialize action probability distribution (APD)
        f_add = np.zeros(self.C.dim_f_add, dtype=np.int32)

        f_conn = np.zeros(self.C.dim_f_conn, dtype=np.int32)

        # determine which nodes are bonded
        bonded_nodes = []
        for bond_type in range(self.n_edge_features):
            bonded_nodes.extend(list(
                np.nonzero(self.edge_features[:, last_node_idx, bond_type])[0]
            ))

        if bonded_nodes:
            degree = len(bonded_nodes)
            v_idx = bonded_nodes[-1]  # idx of node to form bond with
            bond_type_forming = int(
                np.nonzero(self.edge_features[v_idx, last_node_idx, :])[0]
            )

            if degree > 1:
                # if multiple bonds to one node first add bonds one by one
                # (modify `f_conn`)
                f_conn[v_idx, bond_type_forming] = 1

            else:
                # if only bound to one other node, bond and node addition
                # occurs in one move (modify `f_add`)
                f_add[tuple([v_idx] + fv_nonzero_idc + [bond_type_forming])] = 1
        else:
       
            f_add[tuple([0] + fv_nonzero_idc + [0])] = 1

        # concatenate `f_add`, `f_conn`, and `f_term` (`f_term`==0)
        apd = np.concatenate((f_add.ravel(), f_conn.ravel(), np.array([0])))
        return apd

    def get_final_decoding_APD(self):
       
        # initialize action probability distribution (APD)
        f_add = np.zeros(self.C.dim_f_add, dtype=np.int32)

        f_conn = np.zeros(self.C.dim_f_conn, dtype=np.int32)

        # concatenate `f_add`, `f_conn`, and `f_term` (`f_term`==0)
        apd = np.concatenate((f_add.ravel(), f_conn.ravel(), np.array([1])))
        return apd

    def get_graph_state(self):
       
        return self.node_features, self.edge_features

    def get_nonzero_feature_indices(self, node_idx):
      
        fv_idc = get_feature_vector_indices()

        idc = np.nonzero(self.node_features[node_idx])[0]


        segment_idc = [idc[0]]
        for idx, value in enumerate(idc[1:]):
            segment_idc.append(value - fv_idc[idx])

        return segment_idc

    def reorder_nodes(self):
       
        # first remap the node features matrix
        node_features_remapped = np.array(
            [self.node_features[node] for node in self.node_ordering], dtype=np.int32
        )

        # then remap the edge features tensor
        edge_features_rows_done = np.array(
            [self.edge_features[node, :, :] for node in self.node_ordering], dtype=np.int32
        )
        edge_features_remapped = np.array(
            [edge_features_rows_done[:, node, :] for node in self.node_ordering], dtype=np.int32
        )

        self.node_features = node_features_remapped
        self.edge_features = edge_features_remapped

    def pad_graph_representation(self):
    
        node_features_padded = np.zeros((self.C.max_n_nodes,
                                         self.C.n_node_features))
        edge_features_padded = np.zeros((self.C.max_n_nodes,
                                         self.C.max_n_nodes,
                                         self.C.n_edge_features))

        node_features_padded[:self.n_nodes, :] = self.node_features
        edge_features_padded[:self.n_nodes, :self.n_nodes, :] = self.edge_features

        self.node_features = node_features_padded
        self.edge_features = edge_features_padded

    def truncate_graph(self):
      
        last_atom_idx = self.n_nodes - 1

        if self.n_nodes == 1:
            # remove the last atom
            self.node_features[last_atom_idx, :] = 0
            self.n_nodes -= 1
        else:
            # determine how many bonds on the least important atom
            bond_idc = []
            for bond_type in range(self.n_edge_features):
                bond_idc.extend(
                    list(
                        np.nonzero(self.edge_features[:, last_atom_idx, bond_type])[0]
                    )
                )

            degree = len(bond_idc)

            if degree == 1:
 
                self.node_features[last_atom_idx, :] = 0
                self.n_nodes -= 1

            else:  # if degree > 1
               
                bond_idc = bond_idc[-1] 


            self.edge_features[bond_idc, last_atom_idx, :] = 0
            self.edge_features[last_atom_idx, bond_idc, :] = 0


class TrainingGraph(MolecularGraph):
  
    def __init__(self, constants, atom_feature_vector):
        super(TrainingGraph, self).__init__(constants, molecule=False,
                                            node_features=False,
                                            edge_features=False,
                                            atom_feature_vector=False)

        self.n_nodes = int(bool(1 in atom_feature_vector))

        self.node_features = atom_feature_vector.unsqueeze(dim=0)

        self.edge_features = (torch.Tensor)([[[0] * self.n_edge_features]],
                                          device="cuda")

        node_features_padded = torch.zeros((self.C.max_n_nodes,
                                            self.C.n_node_features),
                                           device="cuda")
        edge_features_padded = torch.zeros((self.C.max_n_nodes,
                                            self.C.max_n_nodes,
                                            self.C.n_edge_features),
                                           device="cuda")

        node_features_padded[:self.n_nodes, :] = self.node_features
        edge_features_padded[:self.n_nodes, :self.n_nodes, :] = self.edge_features

        self.node_features = node_features_padded
        self.edge_features = edge_features_padded

    def get_graph_state(self):
       
        node_features_tensor = (torch.Tensor)(self.node_features)
        adjacency_tensor = (torch.Tensor)(self.edge_features)

        return node_features_tensor, adjacency_tensor


class GenerationGraph(MolecularGraph):
  
    def __init__(self, constants, molecule, node_features, edge_features):
        super(GenerationGraph, self).__init__(constants,
                                              molecule=False,
                                              node_features=False,
                                              edge_features=False,
                                              atom_feature_vector=False)

        try:
            self.n_nodes = molecule.GetNumAtoms()
        except AttributeError:
            self.n_nodes = 0

        self.molecule = molecule
        self.node_features = node_features
        self.edge_features = edge_features

    def get_molecule(self):
        
        return self.molecule
