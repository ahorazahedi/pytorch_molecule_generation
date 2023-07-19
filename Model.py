import torch
from parameters import Parameters as C

BIG_POSITIVE = C.big_positive

TORCH_DEVICE = C.device

class MLP(torch.nn.Module):
    
    def __init__(self, in_features, hidden_layer_sizes, out_features, init, dropout_p , activation_function = torch.nn.SELU):
        super(MLP, self).__init__()
    
        fs = [in_features, *hidden_layer_sizes, out_features]
        layers = [self.generate_block(input_linear, out_linear,
                                     activation_function, init,
                                     dropout_p)
                  for input_linear, out_linear in zip(fs, fs[1:])]
        layers = [module for sq in layers for module in sq.children()]
        self.sequence_of_layers = torch.nn.Sequential(*layers)

    def generate_block(self, in_f, out_f, activation, init, dropout_p):
        linear = torch.nn.Linear(in_f, out_f, bias=True)
        return torch.nn.Sequential(linear, activation(), torch.nn.AlphaDropout(dropout_p))

    def forward(self, input):

        return self.sequence_of_layers(input)

class GlobalReadout(torch.nn.Module):
   
    def __init__(self, f_add_elems, f_conn_elems, f_term_elems, mlp1_depth,
                 mlp1_dropout_p, mlp1_hidden_dim, mlp2_depth, mlp2_dropout_p,
                 mlp2_hidden_dim, graph_emb_size, init, max_n_nodes, node_emb_size):

        super(GlobalReadout, self).__init__()

        self.fAddNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_add_elems,
            init=init,
            dropout_p=mlp1_dropout_p , 
            activation_function=torch.nn.ReLU
        )

        self.fConnNet1 = MLP(
            in_features=node_emb_size,
            hidden_layer_sizes=[mlp1_hidden_dim] * mlp1_depth,
            out_features=f_conn_elems,
            init=init,
            dropout_p=mlp1_dropout_p , 
            activation_function=torch.nn.ReLU
            
        )

        self.fAddNet2 = MLP(
            in_features=(max_n_nodes * f_add_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_add_elems * max_n_nodes,
            init=init,
            dropout_p=mlp2_dropout_p , 
            activation_function=torch.nn.ReLU
            
        )

        self.fConnNet2 = MLP(
            in_features=(max_n_nodes * f_conn_elems + graph_emb_size),
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_conn_elems * max_n_nodes,
            init=init,
            dropout_p=mlp2_dropout_p , 
            activation_function=torch.nn.ReLU
            
        )

        self.fTermNet2 = MLP(
            in_features=graph_emb_size,
            hidden_layer_sizes=[mlp2_hidden_dim] * mlp2_depth,
            out_features=f_term_elems,
            init=init,
            dropout_p=mlp2_dropout_p , 
            activation_function=torch.nn.ReLU
        )

    def forward(self, node_level_output, graph_embedding_batch):

        self.fAddNet1 = self.fAddNet1.to(TORCH_DEVICE, non_blocking=True)
        self.fConnNet1 = self.fConnNet1.to(TORCH_DEVICE, non_blocking=True)
        self.fAddNet2 = self.fAddNet2.to(TORCH_DEVICE, non_blocking=True)
        self.fConnNet2 = self.fConnNet2.to(TORCH_DEVICE, non_blocking=True)
        self.fTermNet2 = self.fTermNet2.to(TORCH_DEVICE, non_blocking=True)

        f_add_1 = self.fAddNet1(node_level_output)
        f_conn_1 = self.fConnNet1(node_level_output)

        f_add_1 = f_add_1.to(TORCH_DEVICE, non_blocking=True)
        f_conn_1 = f_conn_1.to(TORCH_DEVICE, non_blocking=True)

    
        f_add_1_size = f_add_1.size()
        f_conn_1_size = f_conn_1.size()
        f_add_1 = f_add_1.view((f_add_1_size[0], f_add_1_size[1] * f_add_1_size[2]))
        f_conn_1 = f_conn_1.view((f_conn_1_size[0], f_conn_1_size[1] * f_conn_1_size[2]))

        f_add_2 = self.fAddNet2(torch.cat((f_add_1, graph_embedding_batch), dim=1).unsqueeze(dim=1))
        f_conn_2 = self.fConnNet2(torch.cat((f_conn_1, graph_embedding_batch), dim=1).unsqueeze(dim=1))
        f_term_2 = self.fTermNet2(graph_embedding_batch)

        f_add_2 = f_add_2.to(TORCH_DEVICE)
        f_conn_2 = f_conn_2.to(TORCH_DEVICE)
        f_term_2 = f_term_2.to(TORCH_DEVICE)

        cat = torch.cat((f_add_2.squeeze(dim=1), f_conn_2.squeeze(dim=1), f_term_2), dim=1)

        return cat  

class GraphGather(torch.nn.Module):

    def __init__(self, node_features, hidden_node_features, out_features,
                 att_depth, att_hidden_dim, att_dropout_p, emb_depth,
                 emb_hidden_dim, emb_dropout_p, init):

        super(GraphGather, self).__init__()

        self.att_nn = MLP(
            in_features=node_features + hidden_node_features,
            hidden_layer_sizes=[att_hidden_dim] * att_depth,
            out_features=out_features,
            init=init,
            dropout_p=att_dropout_p
        )

        self.emb_nn = MLP(
            in_features=hidden_node_features,
            hidden_layer_sizes=[emb_hidden_dim] * emb_depth,
            out_features=out_features,
            init=init,
            dropout_p=emb_dropout_p
        )
        
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, hidden_nodes, input_nodes, node_mask):


        cat = torch.cat((hidden_nodes, input_nodes), dim=2)
        energy_mask = (node_mask == 0).float() * BIG_POSITIVE
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = self.softmax(energies)
        embedding = self.emb_nn(hidden_nodes)

        return torch.sum(attention * embedding, dim=1)

class SummationMPNN(torch.nn.Module):
    def __init__(self, node_features, hidden_node_features, edge_features, message_size, message_passes):

        super(SummationMPNN, self).__init__()

        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.message_size = message_size
        self.message_passes = message_passes

    def message_terms(self, nodes, node_neighbours, edges):
      # Message Passing Function is Here 
      # nodes Shape [batch , node_features]
      # node_nb Shape [batch, max node , number of node features]
      # edges [batch max_node edge feature]
      
        raise NotImplementedError

    def update(self, nodes, messages):
        
      # Message update function is Here 
      # nodes Shape [batch , node_features]
      # messages Shape [batch , number of node features]
        raise NotImplementedError

    def readout(self, hidden_nodes, input_nodes, node_mask):

        #Local readout function
        # hidden_nodes shape [batch , node_feat]
        # input_nodes shape [batch , node_feat]
        # node_mask shape [batch , node_feat]
        raise NotImplementedError

    def forward(self, nodes, edges):
   
        # nodes shape [number of subgraphs , node features , number of nodes]
        # edges shape [number of subgraphs , number_of_node , number_of_node number_of_edge_feat]
        
        adjacency = torch.sum(edges, dim=3)

        (
            edge_batch_batch_idc,
            edge_batch_node_idc,
            edge_batch_nghb_idc,
        ) = adjacency.nonzero(as_tuple=True)

        (node_batch_batch_idc, node_batch_node_idc) = adjacency.sum(-1).nonzero(as_tuple=True)

        same_batch = node_batch_batch_idc.view(-1, 1) == edge_batch_batch_idc
        same_node = node_batch_node_idc.view(-1, 1) == edge_batch_node_idc


        message_summation_matrix = (same_batch * same_node).float()

        edge_batch_edges = edges[edge_batch_batch_idc, edge_batch_node_idc, edge_batch_nghb_idc, :]

        hidden_nodes = torch.zeros(nodes.shape[0], nodes.shape[1], self.hidden_node_features, device=TORCH_DEVICE)
        hidden_nodes[:nodes.shape[0], :nodes.shape[1], :nodes.shape[2]] = nodes.clone()
        node_batch_nodes = hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :]

        for _ in range(self.message_passes):
            edge_batch_nodes = hidden_nodes[edge_batch_batch_idc, edge_batch_node_idc, :]

            edge_batch_nghbs = hidden_nodes[edge_batch_batch_idc, edge_batch_nghb_idc, :]

            message_terms = self.message_terms(edge_batch_nodes,
                                               edge_batch_nghbs,
                                               edge_batch_edges)

            if len(message_terms.size()) == 1: 
                message_terms = message_terms.unsqueeze(0)


            messages = torch.matmul(message_summation_matrix, message_terms)

            node_batch_nodes = self.update(node_batch_nodes, messages)
            hidden_nodes[node_batch_batch_idc, node_batch_node_idc, :] = node_batch_nodes.clone()

        node_mask = adjacency.sum(-1) != 0

        output = self.readout(hidden_nodes, nodes, node_mask)

        return output


class GGNN(SummationMPNN):
 
    def __init__(self, edge_features, enn_depth, enn_dropout_p, enn_hidden_dim,
                 f_add_elems, mlp1_depth, mlp1_dropout_p, mlp1_hidden_dim,
                 mlp2_depth, mlp2_dropout_p, mlp2_hidden_dim, gather_att_depth,
                 gather_att_dropout_p, gather_att_hidden_dim, gather_width,
                 gather_emb_depth, gather_emb_dropout_p, gather_emb_hidden_dim,
                 hidden_node_features, initialization, message_passes,
                 message_size, n_nodes_largest_graph, node_features):

        super(GGNN, self).__init__(node_features, hidden_node_features, edge_features, message_size, message_passes)

        self.n_nodes_largest_graph = n_nodes_largest_graph

        self.msg_nns = torch.nn.ModuleList()
        for _ in range(edge_features):
            self.msg_nns.append(
                MLP(
                    in_features=hidden_node_features,
                    hidden_layer_sizes=[enn_hidden_dim] * enn_depth,
                    out_features=message_size,
                    init=initialization,
                    dropout_p=enn_dropout_p,
                )
            )

        self.gru = torch.nn.GRUCell(
            input_size=message_size, hidden_size=hidden_node_features, bias=True
        )

        self.gather = GraphGather(
            node_features=node_features,
            hidden_node_features=hidden_node_features,
            out_features=gather_width,
            att_depth=gather_att_depth,
            att_hidden_dim=gather_att_hidden_dim,
            att_dropout_p=gather_att_dropout_p,
            emb_depth=gather_emb_depth,
            emb_hidden_dim=gather_emb_hidden_dim,
            emb_dropout_p=gather_emb_dropout_p,
            init=initialization,
        )

        self.APDReadout = GlobalReadout(
            node_emb_size=hidden_node_features,
            graph_emb_size=gather_width,
            mlp1_hidden_dim=mlp1_hidden_dim,
            mlp1_depth=mlp1_depth,
            mlp1_dropout_p=mlp1_dropout_p,
            mlp2_hidden_dim=mlp2_hidden_dim,
            mlp2_depth=mlp2_depth,
            mlp2_dropout_p=mlp2_dropout_p,
            init=initialization,
            f_add_elems=f_add_elems,
            f_conn_elems=edge_features,
            f_term_elems=1,
            max_n_nodes=n_nodes_largest_graph,
        )

    def message_terms(self, nodes, node_neighbours, edges):
        edges_v = edges.view(-1, self.edge_features, 1)
        node_neighbours_v = edges_v * node_neighbours.view(-1, 1, self.hidden_node_features)
        terms_masked_per_edge = [
            edges_v[:, i, :] * self.msg_nns[i](node_neighbours_v[:, i, :])
            for i in range(self.edge_features)
        ]
        return sum(terms_masked_per_edge)

    def update(self, nodes, messages):
        return self.gru(messages, nodes)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        output = self.APDReadout(hidden_nodes, graph_embeddings)

        return output


def create_model():
    net = GGNN(
            f_add_elems=C.dim_f_add_p1,
            edge_features=C.dim_edges[2],
            enn_depth=C.enn_depth,
            enn_dropout_p=C.enn_dropout_p,
            enn_hidden_dim=C.enn_hidden_dim,
            mlp1_depth=C.mlp1_depth,
            mlp1_dropout_p=C.mlp1_dropout_p,
            mlp1_hidden_dim=C.mlp1_hidden_dim,
            mlp2_depth=C.mlp2_depth,
            mlp2_dropout_p=C.mlp2_dropout_p,
            mlp2_hidden_dim=C.mlp2_hidden_dim,
            gather_att_depth=C.gather_att_depth,
            gather_att_dropout_p=C.gather_att_dropout_p,
            gather_att_hidden_dim=C.gather_att_hidden_dim,
            gather_width=C.gather_width,
            gather_emb_depth=C.gather_emb_depth,
            gather_emb_dropout_p=C.gather_emb_dropout_p,
            gather_emb_hidden_dim=C.gather_emb_hidden_dim,
            hidden_node_features=C.hidden_node_features,
            initialization=C.weights_initialization,
            message_passes=C.message_passes,
            message_size=C.message_size,
            n_nodes_largest_graph=C.max_n_nodes,
            node_features=C.dim_nodes[1],
        )

    net = net.to(TORCH_DEVICE, non_blocking=True)

    return net
