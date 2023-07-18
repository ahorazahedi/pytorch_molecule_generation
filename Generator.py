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
from util import reshape_action_prediction  , convert_features_to_atom , sample_action_prediction , convert_graph_to_molecule
class DrugGeneration:
    
    def __init__(self, model : torch.nn.Module, batch_size):
       
        self.start_time = time.time()  # start the timer
        self.batch_size = batch_size
        self.model      = model
        
        self.device = torch.device("cpu")

        # initializes `self.nodes`, `self.edges`, and `self.n_nodes`, which are
        # tensors for keeping track of the batch of graphs
        self.initialize_graph()

        # allocate tensors for finished graphs; these will get filled in gradually
        # as graphs terminate: `self.generated_nodes`, `self.generated_edges`,
        # `self.generated_n_nodes`, `self.generated_likelihoods`, and
        # `self.properly_terminated`
        self.allocate_graph_tensors()

    def sample(self) -> Tuple[list, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        # build the graphs (these are stored as `self` attributes)
        n_generated_graphs = self.build_graphs()
 
       

        # get the time it took to generate graphs
        self.start_time = time.time() - self.start_time
        print(f"Generated {n_generated_graphs} molecules in "
              f"{self.start_time:.4} s" , f"--{n_generated_graphs/self.start_time:4.5} molecules/s")
      

        graphs = [self.graph_to_graph(idx) for idx in range(self.batch_size)]

        final_loglikelihoods = torch.log(
            torch.sum(self.generated_likelihoods, dim=1)[:self.batch_size]
        )
        generated_likelihoods = self.generated_likelihoods[
            self.generated_likelihoods != 0
        ]

        properly_terminated = self.properly_terminated[:self.batch_size]

        return (graphs,
                generated_likelihoods,
                final_loglikelihoods,
                properly_terminated)


    def build_graphs(self) ->  int:
        
        softmax = torch.nn.Softmax(dim=1)

        n_generated_so_far = 0
        generation_round = 0

        
        while n_generated_so_far < self.batch_size:
            
            # predict the APDs for this batch of graphs
            apd = softmax(self.model(self.nodes, self.edges))
            
            # sample the actions from the predicted APDs
            add, conn, term, invalid, likelihoods_just_sampled = self.get_actions(apd)
 
            # indicate (with a 1) the structures which have been properly terminated
            self.properly_terminated[n_generated_so_far:(n_generated_so_far + len(term))] = 1

            # collect the indices for all structures to write (and reset) this round
            termination_idc = torch.cat((term, invalid))
            
            # never write out the dummy graph at index 0
            termination_idc = termination_idc[termination_idc != 0]
            
            # copy the graphs indicated by `terminated_idc` to the tensors for
            # finished graphs (i.e. `generated_{nodes/edges}`)
            n_generated_so_far = self.copy_terminated_graphs(
                termination_idc,
                n_generated_so_far,
                generation_round,
                likelihoods_just_sampled
            )

            # apply actions to all graphs (note: applies dummy actions to
            # terminated graphs, since output will be reset anyways)
            self.apply_actions(add,
                               conn,
                               generation_round,
                               likelihoods_just_sampled)

            # after actions are applied, reset graphs which were set to
            # terminate this round
            self.reset_graphs(termination_idc)

            generation_round += 1

        return n_generated_so_far

    def allocate_graph_tensors(self):
        
        # define tensor shapes
        node_shape = (self.batch_size, *Parameters.dim_nodes)
        edge_shape = (self.batch_size, *Parameters.dim_edges)
        likelihoods_shape = (self.batch_size, Parameters.max_n_nodes * 2)  # the 2 is arbitrary

        # allocate a buffer equal to the size of an extra batch
        n_allocate = self.batch_size * 2

        # create the placeholder tensors:

        # placeholder for node features tensor for all graphs
        self.generated_nodes = torch.zeros((n_allocate, *node_shape[1:]),
                                           dtype=torch.float32,
                                           device=self.device)

        # placeholder for edge features tensor for all graphs
        self.generated_edges = torch.zeros((n_allocate, *edge_shape[1:]),
                                           dtype=torch.float32,
                                           device=self.device)

        # placeholder for number of nodes per graph in all graphs
        self.generated_n_nodes = torch.zeros(n_allocate,
                                             dtype=torch.int8,
                                             device=self.device)


        # placeholder for sampled NLL per action for all graphs
        self.likelihoods = torch.zeros(likelihoods_shape,
                                       device=self.device)

        # placeholder for sampled NLLs per action for all finished graphs
        self.generated_likelihoods = torch.zeros(
            (n_allocate, *likelihoods_shape[1:]),
            device=self.device
        )

        # placeholder for graph termination status (1 == properly terminated,
        # 0 == improper)
        self.properly_terminated = torch.zeros(n_allocate,
                                               dtype=torch.int8,
                                               device=self.device)

    def apply_actions(self, add ,conn , generation_round,likelihoods_sampled ):
        
        def add_new_node(add , generation_round,likelihoods_sampled ):
            add = [idx.long() for idx in add]
            
            
            
            n_node_features = [Parameters.n_atom_types,
                               Parameters.n_formal_charge,
                               Parameters.n_imp_H,
                               Parameters.n_chirality]

            if not Parameters.use_explicit_H and not Parameters.ignore_H:
                if Parameters.use_chirality:
                    (batch, bond_to, atom_type, charge,
                     imp_h, chirality, bond_type, bond_from) = add

                    # add the new nodes to the node features tensors
                    self.nodes[batch, bond_from, atom_type] = 1
                    self.nodes[batch, bond_from, charge + n_node_features[0]] = 1
                    self.nodes[batch, bond_from, imp_h + sum(n_node_features[0:2])] = 1
                    self.nodes[batch, bond_from, chirality + sum(n_node_features[0:3])] = 1
                else:
                    (batch, bond_to, atom_type, charge,
                     imp_h, bond_type, bond_from) = add

                    # add the new nodes to the node features tensors
                    self.nodes[batch, bond_from, atom_type] = 1
                    self.nodes[batch, bond_from, charge + n_node_features[0]] = 1
                    self.nodes[batch, bond_from, imp_h + sum(n_node_features[0:2])] = 1
            elif Parameters.use_chirality:
                (batch, bond_to, atom_type, charge,
                 chirality, bond_type, bond_from) = add

                # add the new nodes to the node features tensors
                self.nodes[batch, bond_from, atom_type] = 1
                self.nodes[batch, bond_from, charge + n_node_features[0]] = 1
                self.nodes[batch, bond_from, chirality + sum(n_node_features[0:2])] = 1
            else:
                (batch, bond_to, atom_type, charge,
                 bond_type, bond_from) = add

                # add the new nodes to the node features tensors
                self.nodes[batch, bond_from, atom_type] = 1
                self.nodes[batch, bond_from, charge + n_node_features[0]] = 1

            # mask dummy edges (self-loops) introduced from adding node to empty graph
            batch_masked = batch[torch.nonzero(self.n_nodes[batch] != 0)]
            bond_to_masked = bond_to[torch.nonzero(self.n_nodes[batch] != 0)]
            bond_from_masked = bond_from[torch.nonzero(self.n_nodes[batch] != 0)]
            bond_type_masked = bond_type[torch.nonzero(self.n_nodes[batch] != 0)]

            # connect newly added nodes to the graphs
            self.edges[batch_masked, bond_to_masked, bond_from_masked, bond_type_masked] = 1
            self.edges[batch_masked, bond_from_masked, bond_to_masked, bond_type_masked] = 1

            # keep track of the newly added node
            self.n_nodes[batch] += 1

            # include the NLLs for the add actions for this generation round
            self.likelihoods[batch, generation_round] = likelihoods_sampled[batch]

        def connect_node(conn , generation_round,
                        likelihoods_sampled ):
            
            # get the action indices
            conn = [idx.long() for idx in conn]
            batch, bond_to, bond_type, bond_from = conn

            # apply the connect actions
            self.edges[batch, bond_from, bond_to, bond_type] = 1
            self.edges[batch, bond_to, bond_from, bond_type] = 1

            # include the NLLs for the connect actions for this generation round
            self.likelihoods[batch, generation_round] = likelihoods_sampled[batch]

        # first applies the "add" action to all graphs in batch (note: does
        # nothing if a graph did not sample "add")
        add_new_node(add, generation_round, likelihoods_sampled)

        # then applies the "connect" action to all graphs in batch (note: does
        # nothing if a graph did not sample "connect")
        connect_node(conn, generation_round, likelihoods_sampled)

    def copy_terminated_graphs(self, terminate_idc ,
                               n_graphs_generated, generation_round,
                               likelihoods_sampled ):
        # number of graphs to be terminated
        self.likelihoods[terminate_idc, generation_round] = likelihoods_sampled[terminate_idc]

        # number of graphs to be terminated
        n_done_graphs = len(terminate_idc)

        # copy the new graphs to the finished tensors
        nodes_local = self.nodes[terminate_idc]
        edges_local = self.edges[terminate_idc]
        n_nodes_local = self.n_nodes[terminate_idc]
        likelihoods_local = self.likelihoods[terminate_idc]

        begin_idx = n_graphs_generated
        end_idx = n_graphs_generated + n_done_graphs
        self.generated_nodes[begin_idx : end_idx] = nodes_local
        self.generated_edges[begin_idx : end_idx] = edges_local
        self.generated_n_nodes[begin_idx : end_idx] = n_nodes_local
        self.generated_likelihoods[begin_idx : end_idx] = likelihoods_local

        n_graphs_generated += n_done_graphs

        return n_graphs_generated

    def add_one_to_init_graph(self):
        self.nodes[0] = torch.ones(([1] + Parameters.dim_nodes),
                                   device=self.device)
        self.edges[0, 0, 0, 0] = 1
        self.n_nodes[0] = 1
        
    def initialize_graph(self):
        node_shape = ([self.batch_size] + Parameters.dim_nodes)
        edge_shape = ([self.batch_size] + Parameters.dim_edges)
        n_nodes_shape = [self.batch_size]
        # initialize tensors
        self.nodes = torch.zeros(node_shape,
                                 dtype=torch.float32,
                                 device=self.device)
        self.edges = torch.zeros(edge_shape,
                                 dtype=torch.float32,
                                 device=self.device)
        self.n_nodes = torch.zeros(n_nodes_shape,
                                   dtype=torch.int8,
                                   device=self.device)
        self.add_one_to_init_graph()
        

    def reset_graphs(self, idc ):
        # define Parameters
        node_shape = ([self.batch_size] + Parameters.dim_nodes)
        edge_shape = ([self.batch_size] + Parameters.dim_edges)
        n_nodes_shape = ([self.batch_size])
        likelihoods_shape = ([self.batch_size] + [Parameters.max_n_nodes * 2])  # the 2 is arbitrary

        # reset the "bad" graphs with zero tensors
        if len(idc) > 0:
            self.nodes[idc] = torch.zeros((len(idc), *node_shape[1:]),
                                          dtype=torch.float32,
                                          device=self.device)
            self.edges[idc] = torch.zeros((len(idc), *edge_shape[1:]),
                                          dtype=torch.float32,
                                          device=self.device)
            self.n_nodes[idc] = torch.zeros((len(idc), *n_nodes_shape[1:]),
                                            dtype=torch.int8,
                                            device=self.device)
            self.likelihoods[idc] = torch.zeros((len(idc), *likelihoods_shape[1:]),
                                                dtype=torch.float32,
                                                device=self.device)

        # create a dummy non-empty graph
        self.nodes[0] = torch.ones(([1] + Parameters.dim_nodes),
                                   device=self.device)
        self.edges[0, 0, 0, 0] = 1
        self.n_nodes[0] = 1

    def get_actions(self, apds ) -> Tuple[torch.Tensor, ...]:
        
        # sample the APD for all graphs in the batch for action indices
        
        f_add_idc, f_conn_idc, f_term_idc, likelihoods = sample_action_prediction(
            apds,
            self.batch_size
        )

        # get indices for the "add" action
        f_add_from = self.n_nodes[f_add_idc[0]]
        f_add_idc = (*f_add_idc, f_add_from)

        # get indices for the "connect" action
        f_conn_from = self.n_nodes[f_conn_idc[0]] - 1
        f_conn_idc = (*f_conn_idc, f_conn_from)

        # get indices for the invalid add and connect actions
        invalid_idc, max_node_idc = self.get_invalid_actions(f_add_idc, f_conn_idc)

        # change "connect to" index for graphs trying to add more than max n nodes
        f_add_idc[5][max_node_idc] = 0

        return f_add_idc, f_conn_idc, f_term_idc, invalid_idc, likelihoods


    def get_invalid_actions(self,
                            f_add_idc ,
                            f_conn_idc ):

        n_max_nodes = Parameters.dim_nodes[0]

        # empty graphs for which "add" action sampled
        f_add_empty_graphs = torch.nonzero(self.n_nodes[f_add_idc[0]] == 0)

        # get invalid indices for when adding a new node to a non-empty graph
        invalid_add_idx_tmp = torch.nonzero(f_add_idc[1] >= self.n_nodes[f_add_idc[0]])
        combined            = torch.cat((invalid_add_idx_tmp, f_add_empty_graphs)).squeeze(1)
        uniques, counts     = combined.unique(return_counts=True)
        invalid_add_idc     = uniques[counts == 1].unsqueeze(dim=1)  # set diff

        # get invalid indices for when adding a new node to an empty graph
        invalid_add_empty_idc = torch.nonzero(f_add_idc[1] != self.n_nodes[f_add_idc[0]])
        combined              = torch.cat((invalid_add_empty_idc, f_add_empty_graphs)).squeeze(1)
        uniques, counts       = combined.unique(return_counts=True)
        invalid_add_empty_idc = uniques[counts > 1].unsqueeze(dim=1)  # set intersection

        # get invalid indices for when adding more nodes than possible
        invalid_madd_idc = torch.nonzero(f_add_idc[5] >= n_max_nodes)

        # get invalid indices for when connecting a node to nonexisting node
        invalid_conn_idc = torch.nonzero(f_conn_idc[1] >= self.n_nodes[f_conn_idc[0]])

        # get invalid indices for when "connecting" a node in a graph with zero nodes
        invalid_conn_nonex_idc = torch.nonzero(self.n_nodes[f_conn_idc[0]] == 0)

        # get invalid indices for when creating self-loops
        invalid_sconn_idc = torch.nonzero(f_conn_idc[1] == f_conn_idc[3])

        # get invalid indices for when attemting to add multiple edges
        invalid_dconn_idc = torch.nonzero(
            torch.sum(self.edges, dim=-1)[f_conn_idc[0].long(),
                                          f_conn_idc[1].long(),
                                          f_conn_idc[-1].long()] == 1
        )
        # only need one invalid index per graph
        invalid_action_idc =torch.unique(
            torch.cat(
                (f_add_idc[0][invalid_add_idc],
                 f_add_idc[0][invalid_add_empty_idc],
                 f_conn_idc[0][invalid_conn_idc],
                 f_conn_idc[0][invalid_conn_nonex_idc],
                 f_conn_idc[0][invalid_sconn_idc],
                 f_conn_idc[0][invalid_dconn_idc],
                 f_add_idc[0][invalid_madd_idc])
            )
        )

        # keep track of invalid indices which require reseting during the final
        # `apply_action()`
        invalid_action_idc_needing_reset = torch.unique(
            torch.cat(
                (invalid_madd_idc, f_add_empty_graphs)
            )
        )

        return invalid_action_idc, invalid_action_idc_needing_reset

    def graph_to_graph(self, idx) -> GenerationGraph:
        
        try:
            # first get the `rdkit.Mol` object corresponding to the selected graph
            mol = convert_graph_to_molecule(self.generated_nodes[idx],
                                self.generated_edges[idx],
                                self.generated_n_nodes[idx])
        except (IndexError, AttributeError):  # raised when graph is empty
            mol = None

        # use the `rdkit.Mol` object, and node and edge features tensors, to get
        # the `GenerationGraph` object
        graph = GenerationGraph(constants=Parameters,
                                molecule=mol,
                                node_features=self.generated_nodes[idx],
                                edge_features=self.generated_edges[idx])
        return graph
