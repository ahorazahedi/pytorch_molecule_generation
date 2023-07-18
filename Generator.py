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
from util import reshape_action_prediction, convert_features_to_atom, sample_from_apd_distribution, convert_graph_to_molecule


class DrugGeneration:

    def __init__(self, model: torch.nn.Module, batch_size):

        self.batch_size = batch_size
        self.model = model

        self.device = torch.device("cpu")

        # initializes `self.nodes`, `self.edges`, and `self.n_nodes`, which are
        # tensors for keeping track of the batch of graphs
        self.initialize_graph()

        # allocate tensors for finished graphs; these will get filled in gradually
        # as graphs terminate: `self.generated_nodes`, `self.generated_edges`,
        # `self.generated_n_nodes`, `self.generated_likelihoods`, and
        # `self.properly_terminated`
        self.allocate_graph_tensors()

    def sample(self):

        # build the graphs (these are stored as `self` attributes)
        n_generated_graphs = self.build_graphs()

        print(f"Generated {n_generated_graphs} molecules")

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

    def build_graphs(self) -> int:
        # Initialize softmax function
        softmax_function = torch.nn.Softmax(dim=1)

        # Set initial values for generated graphs and generation round
        number_of_generated_molecules = 0
        generation_round = 0

        # Keep generating graphs until we reach the desired batch size
        while number_of_generated_molecules < self.batch_size:

            
            # Use the model to predict APDs for the current batch of graphs
            model_output = self.model(self.nodes, self.edges)
            
            # Convert model output to probability distribution using softmax
            apd = softmax_function(model_output)

            """
                node_shape : [batch_size , max_number_of_nodes , node_features]
                edge shape : [batch_size , max_number_of_nodes , max_number_of_nodes , number_of_edge_feature]
                
                dim_f_add   [13, 5, 3, 4]   [max_n_nodes , n_atom_types , n_formal_charge , n_edge_features,]            
                dim_f_conn  [13, 4] [max_n_nodes , n_edge_feature    ]
                dim_f_term  1  
                model_output shape [batch_size , prod(dim_f_add) + prod(dim_f_conn) + 1 ]

            """

            # Sample actions for each graph using the predicted APD
            add_new_atom, connect_node, terminate, invalid, likelihoods_for_this_round = self.get_actions(apd)

            # Mark the structures that have been properly terminated
            # indicate (with a 1) the structures which have been properly terminated
            self.properly_terminated[number_of_generated_molecules:(
                number_of_generated_molecules + len(terminate))] = 1

            # collect the indices for all structures to write (and reset) this round
            # Get indices for all structures to be reset and written to generated graphs in this round
            termination_indices = torch.cat((terminate, invalid))

            # Exclude the dummy graph at index 0 where i put number 1 in some fileds of it
            termination_indices = termination_indices[termination_indices != 0]

            # Copy the terminated graphs to the tensors for finished graphs they start with generated_{node/edge}
            number_of_generated_molecules = self.copy_finalized_graphs_to_generated_graphs(
                termination_indices,
                number_of_generated_molecules,
                generation_round,
                likelihoods_for_this_round
            )

            # apply actions to all graphs (note: applies dummy actions to
            # terminated graphs, since output will be reset anyways)
            self.apply_actions(add_new_atom,
                               connect_node,
                               generation_round,
                               likelihoods_for_this_round)

            # after actions are applied, reset graphs which were set to
            # terminate this round
            self.reset_graphs(termination_indices)

            generation_round += 1

        return number_of_generated_molecules

    def allocate_graph_tensors(self):

        # define tensor shapes
        node_shape = (self.batch_size, *Parameters.dim_nodes)
        edge_shape = (self.batch_size, *Parameters.dim_edges)
        # the 2 is arbitrary
        likelihoods_shape = (self.batch_size, Parameters.max_n_nodes * 2)

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

    def apply_actions(self, add, conn, generation_round, likelihoods_sampled):

        def add_new_node(add, generation_round, likelihoods_sampled):
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
                    self.nodes[batch, bond_from,
                               charge + n_node_features[0]] = 1
                    self.nodes[batch, bond_from, imp_h +
                               sum(n_node_features[0:2])] = 1
                    self.nodes[batch, bond_from, chirality +
                               sum(n_node_features[0:3])] = 1
                else:
                    (batch, bond_to, atom_type, charge,
                     imp_h, bond_type, bond_from) = add

                    # add the new nodes to the node features tensors
                    self.nodes[batch, bond_from, atom_type] = 1
                    self.nodes[batch, bond_from,
                               charge + n_node_features[0]] = 1
                    self.nodes[batch, bond_from, imp_h +
                               sum(n_node_features[0:2])] = 1
            elif Parameters.use_chirality:
                (batch, bond_to, atom_type, charge,
                 chirality, bond_type, bond_from) = add

                # add the new nodes to the node features tensors
                self.nodes[batch, bond_from, atom_type] = 1
                self.nodes[batch, bond_from, charge + n_node_features[0]] = 1
                self.nodes[batch, bond_from, chirality +
                           sum(n_node_features[0:2])] = 1
            else:
                (batch, bond_to, atom_type, charge,
                 bond_type, bond_from) = add

                # add the new nodes to the node features tensors
                self.nodes[batch, bond_from, atom_type] = 1
                self.nodes[batch, bond_from, charge + n_node_features[0]] = 1

            # mask dummy edges (self-loops) introduced from adding node to empty graph
            batch_masked = batch[torch.nonzero(self.n_nodes[batch] != 0)]
            bond_to_masked = bond_to[torch.nonzero(self.n_nodes[batch] != 0)]
            bond_from_masked = bond_from[torch.nonzero(
                self.n_nodes[batch] != 0)]
            bond_type_masked = bond_type[torch.nonzero(
                self.n_nodes[batch] != 0)]

            # connect newly added nodes to the graphs
            self.edges[batch_masked, bond_to_masked,
                       bond_from_masked, bond_type_masked] = 1
            self.edges[batch_masked, bond_from_masked,
                       bond_to_masked, bond_type_masked] = 1

            # keep track of the newly added node
            self.n_nodes[batch] += 1

            # include the NLLs for the add actions for this generation round
            self.likelihoods[batch,
                             generation_round] = likelihoods_sampled[batch]

        def connect_node(conn, generation_round,
                         likelihoods_sampled):

            # get the action indices
            conn = [idx.long() for idx in conn]
            batch, bond_to, bond_type, bond_from = conn

            # apply the connect actions
            self.edges[batch, bond_from, bond_to, bond_type] = 1
            self.edges[batch, bond_to, bond_from, bond_type] = 1

            # include the NLLs for the connect actions for this generation round
            self.likelihoods[batch,
                             generation_round] = likelihoods_sampled[batch]

        # first applies the "add" action to all graphs in batch (note: does
        # nothing if a graph did not sample "add")
        add_new_node(add, generation_round, likelihoods_sampled)

        # then applies the "connect" action to all graphs in batch (note: does
        # nothing if a graph did not sample "connect")
        connect_node(conn, generation_round, likelihoods_sampled)

    def copy_finalized_graphs_to_generated_graphs(self, indices_to_terminate, total_generated_graphs, current_round, sampled_likelihoods):
        # Record the likelihoods of the graphs to be terminated
        self.likelihoods[indices_to_terminate, current_round] = sampled_likelihoods[indices_to_terminate]

        # Number of graphs that will be terminated in this round
        num_terminated_graphs = len(indices_to_terminate)

        # Copy graph components of the terminated graphs
        nodes_to_copy = self.nodes[indices_to_terminate]
        edges_to_copy = self.edges[indices_to_terminate]
        num_nodes_to_copy = self.n_nodes[indices_to_terminate]
        likelihoods_to_copy = self.likelihoods[indices_to_terminate]

        # Determine where to place these in the collection of generated graphs
        start_index = total_generated_graphs
        end_index = total_generated_graphs + num_terminated_graphs

        # Add these graphs to the collection of generated graphs
        self.generated_nodes[start_index: end_index] = nodes_to_copy
        self.generated_edges[start_index: end_index] = edges_to_copy
        self.generated_n_nodes[start_index: end_index] = num_nodes_to_copy
        self.generated_likelihoods[start_index: end_index] = likelihoods_to_copy

        # Update the total count of generated graphs
        total_generated_graphs += num_terminated_graphs

        return total_generated_graphs

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

    def reset_graphs(self, idc):
        # define Parameters
        node_shape = ([self.batch_size] + Parameters.dim_nodes)
        edge_shape = ([self.batch_size] + Parameters.dim_edges)
        n_nodes_shape = ([self.batch_size])
        # the 2 is arbitrary
        likelihoods_shape = ([self.batch_size] + [Parameters.max_n_nodes * 2])

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

    def get_actions(self, apds):
        # Use the softmax output 'apds' to sample actions for each graph in the batch.
        # 'f_add_indices', 'f_conn_indices', and 'f_term_indices' are indices of the add, connect, and terminate actions, respectively.
        # 'likelihoods' contains the likelihoods of these actions.
        f_add_indices, f_conn_indices, f_term_indices, likelihoods = sample_from_apd_distribution(
            apds, self.batch_size)

        # Determine the starting nodes for the "add" actions by indexing 'self.n_nodes' with the first dimension of 'f_add_indices'.
        f_add_from = self.n_nodes[f_add_indices[0]]

        # Extend 'f_add_indices' with the starting node indices for the "add" action.
        f_add_indices = (*f_add_indices, f_add_from)

        # Similar to the "add" action, determine the starting nodes for the "connect" actions.
        # Note that 1 is subtracted here.
        f_conn_from = self.n_nodes[f_conn_indices[0]] - 1

        # Extend 'f_conn_indices' with the starting node indices for the "connect" action.
        f_conn_indices = (*f_conn_indices, f_conn_from)

        # Use the method 'get_invalid_actions' to find invalid "add" and "connect" actions based on the current graph state.
        # The indices of these invalid actions are returned.
        invalid_indices, max_node_indices = self.get_invalid_actions(
            f_add_indices, f_conn_indices)

        # Update the "connect to" index for graphs that are attempting to add more than the maximum allowed number of nodes.
        # For these graphs, the "connect to" index is set to 0.
        f_add_indices[5][max_node_indices] = 0

        # The function returns the indices for "add", "connect", and "terminate" actions, indices of invalid actions,
        # and the likelihoods of each action.
        return f_add_indices, f_conn_indices, f_term_indices, invalid_indices, likelihoods

    def get_invalid_actions(self,
                            f_add_indices,
                            f_conn_indices):
        """
        Consider this Shapes When Try To Read This Code
        add_actions   [batch_size ,  max_n_nodes , n_atom_types , n_formal_charge , n_edge_features,]            
        conn_actions  [batch_size ,  max_n_nodes , n_edge_feature] 
        """

        # Set the maximum number of nodes a graph can have
        n_max_nodes = Parameters.dim_nodes[0]

        # Find indices of empty graphs where an "add" action has been sampled
        f_add_empty_graphs = torch.nonzero(self.n_nodes[f_add_indices[0]] == 0)

        # Identify invalid "add" actions where the "add to" index is larger than the current number of nodes (Non Empty Graphs)
        invalid_add_idx_tmp = torch.nonzero(
            f_add_indices[1] >= self.n_nodes[f_add_indices[0]])
        combined = torch.cat(
            (invalid_add_idx_tmp, f_add_empty_graphs)).squeeze(1)
        uniques, counts = combined.unique(return_counts=True)
        invalid_add_indices = uniques[counts == 1].unsqueeze(dim=1)  # set diff

        # Identify invalid "add" actions when trying to add a new node to an empty graph
        invalid_add_empty_indices = torch.nonzero(
            f_add_indices[1] != self.n_nodes[f_add_indices[0]])
        combined = torch.cat(
            (invalid_add_empty_indices, f_add_empty_graphs)).squeeze(1)
        uniques, counts = combined.unique(return_counts=True)
        invalid_add_empty_indices = uniques[counts > 1].unsqueeze(
            dim=1)  # set intersection

        # Identify actions trying to add more nodes to the graph than the maximum allowed number
        invalid_madd_indices = torch.nonzero(f_add_indices[5] >= n_max_nodes)

        # Find "connect" actions trying to connect to a node that does not exist
        invalid_conn_indices = torch.nonzero(
            f_conn_indices[1] >= self.n_nodes[f_conn_indices[0]])

        # Check for "connect" actions in a graph with zero nodes, which are invalid
        invalid_conn_nonex_indices = torch.nonzero(
            self.n_nodes[f_conn_indices[0]] == 0)

        # Find instances of self-loops, which are invalid
        invalid_sconn_indices = torch.nonzero(
            f_conn_indices[1] == f_conn_indices[3])

        # Identify attempts to add multiple edges between the same nodes, which are invalid
        invalid_dconn_indices = torch.nonzero(
            torch.sum(self.edges, dim=-1)[f_conn_indices[0].long(),
                                          f_conn_indices[1].long(),
                                          f_conn_indices[-1].long()] == 1
        )
        # only need one invalid index per graph
        invalid_action_indices = torch.unique(
            torch.cat(
                (f_add_indices[0][invalid_add_indices],
                 f_add_indices[0][invalid_add_empty_indices],
                 f_conn_indices[0][invalid_conn_indices],
                 f_conn_indices[0][invalid_conn_nonex_indices],
                 f_conn_indices[0][invalid_sconn_indices],
                 f_conn_indices[0][invalid_dconn_indices],
                 f_add_indices[0][invalid_madd_indices])
            )
        )

        # keep track of invalid indices which require reseting during the final
        # `apply_action()`
        invalid_action_indices_needing_reset = torch.unique(
            torch.cat(
                (
                    # Identify actions trying to add more nodes to the graph than the maximum allowed number
                    # Find indices of empty graphs where an "add" action has been sampled
                    invalid_madd_indices, f_add_empty_graphs
                )
            )
        )

        return invalid_action_indices, invalid_action_indices_needing_reset

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
