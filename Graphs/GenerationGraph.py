import numpy as np
from Graphs.MolecularGraph import MolecularGraph



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
