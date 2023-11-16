import os
import copy
import numpy as np
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier
import h5py
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from Parameters import Parameters as C
from MolecularGraph import PreprocessingGraph
from tqdm import tqdm
import argparse


def get_graph(mol):
    
    if mol is not None:
        if not C.use_aromatic_bonds:
            rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
        molecular_graph = PreprocessingGraph(molecule=mol, constants=C)

        return molecular_graph


def calculate_reversing_decode_route_length(molecular_graph):
    
    return molecular_graph.get_n_edges() + 2


def get_dataset_dims():

    dims = {}
    dims["nodes"] = C.dim_nodes
    dims["edges"] = C.dim_edges
    dims["APDs"] = [np.prod(C.dim_f_add) + np.prod(C.dim_f_conn) + 1]
    return dims


def create_datasets(hdf_file, max_length, dataset_name_list, dims):
    ds = {}

    for ds_name in dataset_name_list:
        ds[ds_name] = hdf_file.create_dataset(ds_name,
                                              (max_length, *dims[ds_name]),
                                              chunks=True,
                                              dtype=np.dtype("int8"))

    return ds
def save_group(dataset_dict, data_subgraphs, data_APDs, n_SGs, init_idx):

    nodes = np.array([graph_tuple[0] for graph_tuple in data_subgraphs])
    edges = np.array([graph_tuple[1] for graph_tuple in data_subgraphs])
    APDs = np.array(data_APDs)

    end_idx = init_idx + n_SGs 

    dataset_dict["nodes"][init_idx:end_idx] = nodes
    dataset_dict["edges"][init_idx:end_idx] = edges
    dataset_dict["APDs"][init_idx:end_idx] = APDs

    return dataset_dict


def get_graph(mol):
    
    if mol is not None:
        if not C.use_aromatic_bonds:
            rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
        molecular_graph = PreprocessingGraph(molecule=mol, constants=C)

        return molecular_graph

def generate_decoding_states(molecular_graph, subgraph_idx):

    molecular_graph = copy.deepcopy(molecular_graph)

    if subgraph_idx != 0:
        
        for _ in range(1, subgraph_idx):
            molecular_graph.truncate_graph()
            
        decoding_APD = molecular_graph.get_decoding_APD()
        molecular_graph.truncate_graph()
        
        X, E = molecular_graph.get_graph_state()
        
    elif subgraph_idx == 0:
        
        decoding_APD = molecular_graph.get_final_decoding_APD()
        
        X, E = molecular_graph.get_graph_state()

    else:
        raise ValueError("`subgraph_idx` not a valid value.")

    decoding_graph = [X, E]
    return decoding_graph, decoding_APD
def group_subgraphs(init_idx, molecule, dataset_dict):
   
    data_subgraphs = []        
    data_APDs = []           
 
    molecular_graph_generator = get_graph(molecule)

    molecules_processed = 0 
    
    for graph in [molecular_graph_generator]:
      
        molecules_processed += 1

        n_SGs = calculate_reversing_decode_route_length(molecular_graph=graph)

        for new_SG_idx in range(n_SGs):  
            
            SG, APD = generate_decoding_states(molecular_graph=graph,
                                                   subgraph_idx=new_SG_idx)
            data_subgraphs.append(SG)
            data_APDs.append(APD)
          
    dataset_dict = save_group(dataset_dict=dataset_dict,
                              n_SGs=n_SGs,
                              data_subgraphs=data_subgraphs,
                              data_APDs=data_APDs,
                              init_idx=init_idx)

    len_data_subgraphs = len(data_subgraphs)
    return molecules_processed, dataset_dict, len_data_subgraphs


def execute_file(file_name ='./data/train.smi_chunk_1.smi' ):
    
    h5_file_name = file_name[:-3] + "h5"

    with open(file_name) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
        
    smi_file.close()

    molecule_set =  SmilesMolSupplier(file_name, sanitize=True, nameColumn=-1, titleLine=has_header)
    
    number_of_molecule = len(molecule_set)

    n_subgraphs = 0  

    molecular_graph_generator = map(get_graph, molecule_set)

    for molecular_graph in tqdm(molecular_graph_generator , total=number_of_molecule):
        n_SGs = calculate_reversing_decode_route_length(molecular_graph=molecular_graph)
        n_subgraphs += n_SGs

    dataset_names = ["nodes", "edges", "APDs"]

    dims = get_dataset_dims()

    if os.path.exists(h5_file_name):
            print("Chunk File Already exist Removing Previous Chunk File")
            os.remove(h5_file_name)
            
    with h5py.File(h5_file_name, "w") as hdf_file:
        print(f"Creating {h5_file_name} File To Store APDs")
        
        ds = create_datasets(hdf_file=hdf_file,
                                max_length=n_subgraphs,
                                dataset_name_list=dataset_names,
                                dims=dims)
        
        dataset_size = 0  

        for init_idx in tqdm(range(0, number_of_molecule)):

            (final_molecule_idx, ds, len_data_subgraphs) = group_subgraphs(init_idx=init_idx,
                                                molecule=molecule_set[init_idx],
                                                dataset_dict=ds,                                           
                                            )
            
            dataset_size += len_data_subgraphs
            

def main():
    parser = argparse.ArgumentParser(description='Split an SMI file into three parts.')
    parser.add_argument('--file', help='Path to the SMI file to split')

    args = parser.parse_args()
    
    execute_file(args.file)
 
if __name__ == '__main__':
    main()