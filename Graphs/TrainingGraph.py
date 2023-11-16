import numpy as np
import torch
from Parameters import Parameters as C
from Graphs.MolecularGraph import MolecularGraph


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
