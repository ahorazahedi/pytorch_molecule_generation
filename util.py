# load general packages and functions
import csv
import numpy as np
import torch
import rdkit


# load program-specific functions
from Parameters import Parameters as C
from typing import Tuple
from Parameters import Parameters
from rdkit.Chem import MolToSmiles


def reshape_action_prediction(apds, batch_size):
    
    """
        This Function get Model Output and Split Each Action For Each item In Batch size
    """

    # get shapes of "add" and "connect" actions
    f_add_shape = (batch_size, *Parameters.dim_f_add)
    f_conn_shape = (batch_size, *Parameters.dim_f_conn)

    # dim_f_add   [13, 5, 3, 4]   max_n_nodes , n_atom_types , n_formal_charge , n_edge_features,            
    # dim_f_conn  [13, 4] max_n_nodes , n_edge_feature    

    
    # get len of flattened segment of APD corresponding to "add" action
    f_add_size = np.prod(Parameters.dim_f_add)


    # split apd from model output
    # reshape the various APD components  
    f_add = torch.reshape(apds[:, :f_add_size], f_add_shape)
    f_conn = torch.reshape(apds[:, f_add_size:-1], f_conn_shape)
    f_term = apds[:, -1]
    
    # f_add   [6,13, 5, 3, 4]   batch_size , max_n_nodes , n_atom_types , n_formal_charge , n_edge_features,            
    # f_conn  [6,13, 4] batch_size ,max_n_nodes , n_edge_feature  
    # f_term  [6,1]     batch_size , 1
    
   
    return f_add, f_conn, f_term


def sample_from_apd_distribution(apds, batch_size):

    # Creating a Multinomial distribution object, which is used to model the probability of counts of different outcomes.
    # Here, the distribution is parameterized by the softmax output 'apds'. 
    # The shape of 'apds' is expected to be [batch_size, prod(add), prod(conn), prod(terminae)].
    action_probability_distribution = torch.distributions.Multinomial(1, probs=apds)

    # Drawing a sample from the action_probability_distribution.
    # The result is a one-hot tensor 'apd_one_hot' with the same shape as 'apds'.
    apd_one_hot = action_probability_distribution.sample()

    # Reshaping 'apd_one_hot' into three separate tensors: 'f_add', 'f_conn', and 'f_term',
    # each representing a different potential action.
    f_add, f_conn, f_term = reshape_action_prediction(apd_one_hot, batch_size)

    # Extracting the probabilities of the actions represented by 'apd_one_hot' from the softmax output 'apds'.
    # The resulting tensor 'likelihoods' represents the likelihood of each action, and its shape is [batch_size].
    likelihoods = apds[apd_one_hot == 1]    

    # Finding the indices of the non-zero elements in 'f_add', 'f_conn', and 'f_term'.
    # These indices correspond to the actions predicted by the model.
    # The 'as_tuple=True' argument changes the output to a tuple of 1D tensors, 
    # which is more convenient for indexing multi-dimensional tensors.
    add_indicies = torch.nonzero(f_add , as_tuple=True) 
    conn_indicies = torch.nonzero(f_conn, as_tuple=True)

    # The '.view(-1)' operation reshapes the output of 'torch.nonzero(f_term)' into a 1D tensor.
    term_indicies = torch.nonzero(f_term).view(-1)

    return add_indicies, conn_indicies, term_indicies, likelihoods


def convert_graph_to_molecule(node_features,
                              edge_features,
                              n_nodes) -> rdkit.Chem.Mol:

    # create empty editable `rdkit.Chem.Mol` object
    molecule = rdkit.Chem.RWMol()
    node_to_idx = {}

    # add atoms to editable mol object
    for node_idx in range(n_nodes):
        atom_to_add = convert_node_to_atom_type_based_on_features(node_idx, node_features)
        molecule_idx = molecule.AddAtom(atom_to_add)
        node_to_idx[node_idx] = molecule_idx

    # add bonds to atoms in editable mol object; to not add the same bond twice
    # (which leads to an error), mask half of the edge features beyond diagonal
    n_max_nodes = Parameters.dim_nodes[0]
    edge_mask = torch.triu(
        torch.ones((n_max_nodes, n_max_nodes), device=Parameters.device),
        diagonal=1
    )
    edge_mask = edge_mask.view(n_max_nodes, n_max_nodes, 1)
    edges_idc = torch.nonzero(edge_features * edge_mask)

    for node_idx1, node_idx2, bond_idx in edges_idc:
        molecule.AddBond(
            node_to_idx[node_idx1.item()],
            node_to_idx[node_idx2.item()],
            Parameters.int_to_bondtype[bond_idx.item()],
        )

    try:  # convert editable mol object to non-editable mol object
        molecule.GetMol()
    except AttributeError:  # will throw an error if molecule is `None`
        pass

    if Parameters.ignore_H and molecule:
        try:  # correct for ignored Hs
            rdkit.Chem.SanitizeMol(molecule)
        except ValueError:  # throws exception if molecule is too ugly to correct
            pass

    return molecule


def convert_node_to_atom_type_based_on_features(node_idx, node_features):

    # get all the nonzero indices in the specified node feature vector
    nonzero_idc = torch.nonzero(node_features[node_idx])

    # determine atom symbol
    atom_idx = nonzero_idc[0]
    atom_type = Parameters.atom_types[atom_idx]

    # initialize atom
    new_atom = rdkit.Chem.Atom(atom_type)

    # determine formal charge
    fc_idx = nonzero_idc[1] - Parameters.n_atom_types
    formal_charge = Parameters.formal_charge[fc_idx]

    new_atom.SetFormalCharge(formal_charge)  # set property

    # determine number of implicit Hs (if used)
    if not Parameters.use_explicit_H and not Parameters.ignore_H:
        total_num_h_idx = (nonzero_idc[2] -
                           Parameters.n_atom_types -
                           Parameters.n_formal_charge)
        total_num_h = Parameters.imp_H[total_num_h_idx]

        new_atom.SetUnsignedProp("_TotalNumHs", total_num_h)  # set property
    elif Parameters.ignore_H:
        # Hs will be set with structure is "sanitized" (corrected) later
        # in `mol_to_graph()`
        pass

    # determine chirality (if used)
    if Parameters.use_chirality:
        cip_code_idx = (
            nonzero_idc[-1]
            - Parameters.n_atom_types
            - Parameters.n_formal_charge
            - (not Parameters.use_explicit_H and not
               Parameters.ignore_H) * Parameters.n_imp_H
        )
        cip_code = Parameters.chirality[cip_code_idx]
        new_atom.SetProp("_CIPCode", cip_code)  # set property

    return new_atom


def write_molecules(molecules,
                    name,
                    ):

    # save molecules as SMILE
    smi_filename = f"generation/{name}.smi"

    (fraction_valid,
     validity_tensor,
     uniqueness_tensor) = write_graphs_to_smi(smi_filename=smi_filename,
                                              molecular_graphs_list=molecules,
                                            )

    return fraction_valid, validity_tensor, uniqueness_tensor


def write_graphs_to_smi(smi_filename,molecular_graphs_list):

    validity_tensor = torch.zeros(len(molecular_graphs_list),
                                  device=Parameters.device)
    uniqueness_tensor = torch.ones(len(molecular_graphs_list),
                                   device=Parameters.device)
    smiles = []

    with open(smi_filename, "w") as smi_file:
        smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(smi_file)

        for idx, molecular_graph in enumerate(molecular_graphs_list):

            mol = molecular_graph.get_molecule()
            try:
                mol.UpdatePropertyCache(strict=False)
                rdkit.Chem.SanitizeMol(mol)
                smi_writer.write(mol)
                current_smiles = MolToSmiles(mol)
                validity_tensor[idx] = 1
                if current_smiles in smiles:
                    uniqueness_tensor[idx] = 0
                smiles.append(current_smiles)
            except (ValueError, RuntimeError, AttributeError):
                # molecule cannot be written to file (likely contains unphysical
                # aromatic bond(s) or an unphysical valence), so put placeholder
               
                # `validity_tensor` remains 0
                smi_writer.write(rdkit.Chem.MolFromSmiles("[Xe]"))

        smi_writer.close()

    fraction_valid = torch.sum(validity_tensor, dim=0) / len(validity_tensor)

    return fraction_valid, validity_tensor, uniqueness_tensor
