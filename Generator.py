# load general packages and functions
import time
from typing import Tuple
import numpy as np
from tqdm import tqdm
import torch
import rdkit
from pprint import pprint

from Parameters import Parameters
from MolecularGraph import GenerationGraph
from tabulate import tabulate


class DrugGeneration:

    def __init__(self, model=None, batch_size=7, constant_parameters=Parameters) -> None:

        self.model = model
        self.batch_size = batch_size
        self.constant_parameters = constant_parameters

        self.device = torch.device('cpu')

        self.node_shape = ([self.batch_size] +
                           self.constant_parameters.dim_nodes)
        self.edge_shape = ([self.batch_size] +
                           self.constant_parameters.dim_edges)
        self.n_nodes_shape = [self.batch_size]
        
        self.likelihoods_shape = ([self.batch_size] + [self.constant_parameters.max_n_nodes * 2])  # the 2 is arbitrary

        summary_data = [
            ["likelihoods_shape" , self.likelihoods_shape] ,  
            ["node shape", self.node_shape],
            ["edge_shape", self.edge_shape],
            ["n node shape", self.n_nodes_shape],
            ["Batch Size", self.batch_size],
        ]

        print(tabulate(summary_data, tablefmt="fancy_grid"))

    def set_one_value(self):
        # Set All Row Zero Values to 1
        
        self.nodes[0] = torch.ones(
            [1] + self.constant_parameters.dim_nodes, device=self.device, dtype=torch.float32)
        # set edge one bound type to 1
        self.edges[0, 0, 0, 0] = 1

        # say that i have just one Molecule With one Atom
        self.n_nodes[0] = 1
        
        
        
    def init_graphs(self):
        """
            init Empyt Graphs With Zero Values 
            set first item of each tensor to 1 for model input value 
            shapes are

            nodes : ([self.batch_size] + constants.dim_nodes)
            edges : ([self.batch_size] + constants.dim_edges)
            n_nodes :  [self.batch_size]
        """
        self.nodes = torch.zeros(
            self.node_shape, dtype=torch.float32, device=self.device)
        self.edges = torch.zeros(
            self.edge_shape, dtype=torch.float32, device=self.device)
        self.n_nodes = torch.zeros(
            self.n_nodes_shape, dtype=torch.int8, device=self.device)
        self.set_one_value()
        


if __name__ == "__main__":
    # pprint(Parameters)
    d = DrugGeneration()
    d.init_graphs()
