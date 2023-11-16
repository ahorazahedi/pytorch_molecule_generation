"""
Combines preprocessed HDF files. Useful when preprocessing large datasets, as
one can split the `{split}.smi` into multiple files (and directories), preprocess
them separately, and then combine using this script.

To use script, modify the variables below to automatically create a list of
paths **assuming** HDFs were created with the following directory structure:
 data/
  |-- {dataset}_1/
  |-- {dataset}_2/
  |-- {dataset}_3/
  |...
  |-- {dataset}_{n_dirs}/

The variables are also used in setting the dimensions of the HDF datasets later on.

If directories were not named as above, then simply replace `path_list` below
with a list of the paths to all the HDFs to combine.

Then, run:
python combine_HDFs.py
"""
import numpy as np
import h5py

def get_dims() -> dict:
    """
    Gets the dims corresponding to the three datasets in each preprocessed HDF
    file: "nodes", "edges", and "APDs".
    """
    dims = {}
    dims["nodes"] = [max_n_nodes, n_atom_types + n_formal_charges]
    dims["edges"] = [max_n_nodes, max_n_nodes, n_bond_types]
    dim_f_add     = [max_n_nodes, n_atom_types, n_formal_charges, n_bond_types]
    dim_f_conn    = [max_n_nodes, n_bond_types]
    dims["APDs"]  = [np.prod(dim_f_add) + np.prod(dim_f_conn) + 1]

    return dims

def get_total_n_subgraphs(paths : list) -> int:
    """
    Gets the total number of subgraphs saved in all the HDF files in the `paths`,
    where `paths` is a list of strings containing the path to each HDF file we want
    to combine.
    """
    total_n_subgraphs = 0
    for path in paths:
        print("path:", path)
        hdf_file           = h5py.File(path, "r")
        nodes              = hdf_file.get("nodes")
        n_subgraphs        = nodes.shape[0]
        total_n_subgraphs += n_subgraphs
        hdf_file.close()

    return total_n_subgraphs

def main(paths : list) -> None:
    """
    Combine many small HDF files (their paths defined in `paths`) into one large HDF file.
    """
    total_n_subgraphs = get_total_n_subgraphs(paths)
    dims              = get_dims()

    print(f"* Creating HDF file to contain {total_n_subgraphs} subgraphs")
    new_hdf_file = h5py.File(f"data/{dataset}/{split}.h5", "a")
    new_dataset_nodes = new_hdf_file.create_dataset("nodes",
                                                    (total_n_subgraphs, *dims["nodes"]),
                                                    dtype=np.dtype("int8"))
    new_dataset_edges = new_hdf_file.create_dataset("edges",
                                                    (total_n_subgraphs, *dims["edges"]),
                                                    dtype=np.dtype("int8"))
    new_dataset_APDs  = new_hdf_file.create_dataset("APDs",
                                                    (total_n_subgraphs, *dims["APDs"]),
                                                    dtype=np.dtype("int8"))

    print("* Combining data from smaller HDFs into a new larger HDF.")
    init_index = 0
    for path in paths:
        print("path:", path)
        hdf_file = h5py.File(path, "r")

        nodes = hdf_file.get("nodes")
        edges = hdf_file.get("edges")
        APDs  = hdf_file.get("APDs")

        n_subgraphs = nodes.shape[0]

        new_dataset_nodes[init_index:(init_index + n_subgraphs)] = nodes
        new_dataset_edges[init_index:(init_index + n_subgraphs)] = edges
        new_dataset_APDs[init_index:(init_index + n_subgraphs)]  = APDs

        init_index += n_subgraphs
        hdf_file.close()

    new_hdf_file.close()



if __name__ == "__main__":
    # combine the HDFs defined in `path_list`

    # set variables
    dataset          = "ChEMBL"
    n_atom_types     = 15       # number of atom types used in preprocessing the data
    n_formal_charges = 3        # number of formal charges used in preprocessing the data
    n_bond_types     = 3        # number of bond types used in preprocessing the data
    max_n_nodes      = 40       # maximum number of nodes in the data

    # combine the training files
    n_dirs           = 12       # how many times was `{split}.smi` split?
    split            = "train"  # train, test, or valid
    path_list        = [f"data/{dataset}_{i}/{split}.h5" for i in range(0, n_dirs)]
    main(path_list)

    # combine the test files
    n_dirs           = 4       # how many times was `{split}.smi` split?
    split            = "test"  # train, test, or valid
    path_list        = [f"data/{dataset}_{i}/{split}.h5" for i in range(0, n_dirs)]
    main(path_list)

    # combine the validation files
    n_dirs           = 2        # how many times was `{split}.smi` split?
    split            = "valid"  # train, test, or valid
    path_list        = [f"data/{dataset}_{i}/{split}.h5" for i in range(0, n_dirs)]
    main(path_list)

    print("Done.", flush=True)
