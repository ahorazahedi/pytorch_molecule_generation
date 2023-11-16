import os
import copy

import numpy as np
import rdkit
from rdkit.Chem.rdmolfiles import SmilesMolSupplier
import h5py
from tqdm import tqdm


from Parameters import Parameters as C

from Graphs.GenerationGraph import PreprocessingGraph

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


def create_datasets(hdf_file, max_length, dataset_name_list, dims):
    ds = {}

    for ds_name in dataset_name_list:
        ds[ds_name] = hdf_file.create_dataset(ds_name,
                                              (max_length, *dims[ds_name]),
                                              chunks=True,
                                              dtype=np.dtype("int8"))

    return ds



def load_smiles_file(path):

    # checking For Existing Header in Smi File
    with open(path) as smi_file:
        first_line = smi_file.readline()
        has_header = bool("SMILES" in first_line)
    smi_file.close()

    # read file
    molecule_set = SmilesMolSupplier(path, sanitize=True, nameColumn=-1, titleLine=has_header)

    return molecule_set


def create_HDF_file(path):

    molecule_set = load_smiles_file(path)

    number_of_molecule = len(molecule_set)
    
    total_number_of_subgraphs = calculate_number_subgraphs_in_molecule(molecule_set=molecule_set)

    dataset_names = ["nodes", "edges", "APDs"]
    dims = get_dataset_dims()

    if os.path.exists(f"{path[:-3]}h5.chunked"):
        print("Chunk File Already exist Removing Previous Chunk File")
        os.remove(f"{path[:-3]}h5.chunked")
        
        
    with h5py.File(f"{path[:-3]}h5.chunked", "a") as hdf_file:
        print("Creating HDF File To Store APDs")
        
        ds = create_datasets(hdf_file=hdf_file,
                                max_length=total_number_of_subgraphs,
                                dataset_name_list=dataset_names,
                                dims=dims)
        

        dataset_size = 0  

        print("Start Looping Over Molecules")
        for init_idx in tqdm(range(0, number_of_molecule)):
    
            (final_molecule_idx, ds, len_data_subgraphs) = group_subgraphs(init_idx=init_idx,
                                                molecule=molecule_set[init_idx],
                                                dataset_dict=ds,                                           
                                           )
            
            dataset_size += len_data_subgraphs
     
 
    print("Saving Chunk File As Final File")
    with h5py.File(f"{path[:-3]}h5.chunked", "r", swmr=True) as chunked_file:
        keys = list(chunked_file.keys())
        data = [chunked_file.get(key)[:] for key in keys]
        data_zipped = tuple(zip(data, keys))

        with h5py.File(f"{path[:-3]}h5", "w") as unchunked_file:
            for d, k in tqdm(data_zipped):
                unchunked_file.create_dataset(k, chunks=None, data=d, dtype=np.dtype("int8"))

    print("Removing Temperatory Chunk File")
    os.remove(f"{path[:-3]}h5.chunked")
    return f"{path[:-3]}h5"
 

def get_dataset_dims():

    dims = {}
    dims["nodes"] = C.dim_nodes
    dims["edges"] = C.dim_edges
    dims["APDs"] = [np.prod(C.dim_f_add) + np.prod(C.dim_f_conn) + 1]
    return dims


def get_graph(mol):
    
    if mol is not None:
        if not C.use_aromatic_bonds:
            rdkit.Chem.Kekulize(mol, clearAromaticFlags=True)
        molecular_graph = PreprocessingGraph(molecule=mol, constants=C)

        return molecular_graph



def calculate_reversing_decode_route_length(molecular_graph):
    
    return molecular_graph.get_n_edges() + 2


def calculate_number_subgraphs_in_molecule(molecule_set):
    n_subgraphs = 0  
    
    molecular_graph_generator = map(get_graph, molecule_set)

    for molecular_graph in molecular_graph_generator:
        n_SGs = calculate_reversing_decode_route_length(molecular_graph=molecular_graph)
        n_subgraphs += n_SGs

    return n_subgraphs



def load_datasets(hdf_file, dataset_name_list):

    ds = {} 
    for ds_name in dataset_name_list:
        ds[ds_name] = hdf_file.get(ds_name)

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


 
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    create_HDF_file(path=args.path)