import torch
import h5py


class HDFDataset(torch.utils.data.Dataset):

    def __init__(self, path):

        self.path = path

        hdf_file = h5py.File(self.path, "r+", swmr=True)
   
        self.nodes = hdf_file.get("nodes")
        self.edges = hdf_file.get("edges")
        self.apds = hdf_file.get("APDs")

        self.n_subgraphs = self.nodes.shape[0]

    def __getitem__(self, idx):

        nodes_i = torch.from_numpy(self.nodes[idx]).type(torch.float32)
        edges_i = torch.from_numpy(self.edges[idx]).type(torch.float32)
        apd_i = torch.from_numpy(self.apds[idx]).type(torch.float32)

        return (nodes_i, edges_i, apd_i)

    def __len__(self):
        return self.n_subgraphs
