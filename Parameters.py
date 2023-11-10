
from collections import namedtuple 
import pprint

import torch
import numpy as np
from rdkit.Chem.rdchem import BondType
 

Molecular_Parameters = {
    'use_aromatic_bonds' : True  , 
    "atom_types": ["C", "N", "O", "S", "Cl"],
    "formal_charge": [-1, 0, 1],
    "imp_H": [0, 1, 2, 3],
    "chirality": ["None", "R", "S"],
    "generation_epoch": 30,
    "n_samples": 2000,  #5000,
    "n_workers": 2,
    "restart": False,
    "max_n_nodes": 13,
    "job_type": "train",
    "sample_every": 10,
    
    "use_canon": True,
    "use_chirality": False,
    "use_explicit_H": False,
    "ignore_H": True,
    
}

Learning_Parameters = {
    "batch_size": 1000,
    "block_size": 100000,
    "epochs": 100,
    "init_lr": 1e-4,
    "lr_ramp_up_minibatches": 500,
    "lrdf": 0.9999,
    "lrdi": 100,
    "min_rel_lr": 5e-2,
    "max_rel_lr": 10,
    "ramp_up_lr": False,
    "weights_initialization": "uniform",
    "weight_decay": 0.0,

}

Model_Parameters = {
    "enn_depth": 4,
    "enn_dropout_p": 0.0,
    "enn_hidden_dim": 250,
    "mlp1_depth": 4,
    "mlp1_dropout_p": 0.0,
    "mlp1_hidden_dim": 500,
    "mlp2_depth": 4,
    "mlp2_dropout_p": 0.0,
    "mlp2_hidden_dim": 500,
    "gather_att_depth": 4,
    "gather_att_dropout_p": 0.0,
    "gather_att_hidden_dim": 250,
    "gather_emb_depth": 4,
    "gather_emb_dropout_p": 0.0,
    "gather_emb_hidden_dim": 250,
    "gather_width": 100,
    "hidden_node_features": 100,
    "message_passes": 3,
    "message_size": 100,
}




def calculate_feat_dim(parameters):
 
    n_atom_types = len(parameters["atom_types"])
    n_formal_charge = len(parameters["formal_charge"])
    n_numh = int(
        not parameters["use_explicit_H"]
        and not parameters["ignore_H"]
    ) * len(parameters["imp_H"])
    n_chirality = int(parameters["use_chirality"]) * len(parameters["chirality"])

    return n_atom_types, n_formal_charge, n_numh, n_chirality


def calculate_tensor_dim(n_atom_types,
                          n_formal_charge,
                          n_num_h,
                          n_chirality,
                          n_node_features,
                          n_edge_features,
                          parameters):

    max_nodes = parameters["max_n_nodes"]

    dim_nodes = [max_nodes, n_node_features]

    dim_edges = [max_nodes, max_nodes, n_edge_features]

    if parameters["use_chirality"]:
        if parameters["use_explicit_H"] or parameters["ignore_H"]:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_chirality,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_chirality,
                n_edge_features,
            ]
    else:
        if parameters["use_explicit_H"] or parameters["ignore_H"]:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                parameters["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_edge_features,
            ]

    dim_f_conn = [parameters["max_n_nodes"], n_edge_features]

    dim_f_term = 1

    return dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term



def collect_global_constants(parameters):

    if parameters["use_explicit_H"] and parameters["ignore_H"]:
        raise ValueError(
            "use explicit H's and ignore H's At Same Time :/ "
    
        )
    

    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    
    if parameters["use_aromatic_bonds"]:
        bondtype_to_int[BondType.AROMATIC] = 3
    
    int_to_bondtype = dict(map(reversed, bondtype_to_int.items()))
    
    n_edge_features = len(bondtype_to_int)

    n_atom_types, n_formal_charge, n_imp_H, n_chirality = calculate_feat_dim(parameters)
    
    n_node_features = n_atom_types + n_formal_charge + n_imp_H + n_chirality

    dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term = calculate_tensor_dim(
        n_atom_types,
        n_formal_charge,
        n_imp_H,
        n_chirality,
        n_node_features,
        n_edge_features,
        parameters,
    )
    
    dim_f_add_p0 = np.prod(dim_f_add[:])
    dim_f_add_p1 = np.prod(dim_f_add[1:])
    dim_f_conn_p0 = np.prod(dim_f_conn[:])
    dim_f_conn_p1 = np.prod(dim_f_conn[1:])

    constants_dict = {
        "big_negative": -1e6,
        "big_positive": 1e6,
        "bondtype_to_int": bondtype_to_int,
        "int_to_bondtype": int_to_bondtype,
        "n_edge_features": n_edge_features,
        "n_atom_types": n_atom_types,
        "n_formal_charge": n_formal_charge,
        "n_imp_H": n_imp_H,
        "n_chirality": n_chirality,
        "n_node_features": n_node_features,
        "dim_nodes": dim_nodes,
        "dim_edges": dim_edges,
        "dim_f_add": dim_f_add,
        "dim_f_conn": dim_f_conn,
        "dim_f_term": dim_f_term,
        "dim_f_add_p0": dim_f_add_p0,
        "dim_f_add_p1": dim_f_add_p1,
        "dim_f_conn_p0": dim_f_conn_p0,
        "dim_f_conn_p1": dim_f_conn_p1,
        "device" : torch.device("cpu")
    }
    

    constants_dict.update(parameters)
    
    
    Constants = namedtuple("CONSTANTS", sorted(constants_dict))
    constants = Constants(**constants_dict)
    
    
    del constants_dict["bondtype_to_int"]
    del constants_dict["int_to_bondtype"]
    
    
    from tabulate import tabulate
    table_data = [(key, value) for key, value in constants_dict.items()]
    table = tabulate(table_data, headers=["Key", "Value"], tablefmt="fancy_grid")
    print(table)

    return constants

Default_Parameters = {**Molecular_Parameters , **Model_Parameters , **Learning_Parameters}

Parameters = collect_global_constants(parameters=Default_Parameters,
                                     )
# if __name__ == "__main__":
#     from tabulate import tabulate
#     table_data = [(key, value) for key, value in Parameters.items()]
#     table = tabulate(table_data, headers=["Key", "Value"], tablefmt="fancy_grid")
#     print(table)

    