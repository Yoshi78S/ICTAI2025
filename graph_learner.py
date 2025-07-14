import torch
import torch.nn as nn
from layers import Attentive, GCNConv_dense
from Utils.Utils import *
import torch.nn.functional as F
import scipy.sparse as sp
import time
from Params import args


class UserItemGraph(nn.Module):
    def __init__(self, n_attlayers, n_mlplayers, hidden_size, user_isize, item_isize, k, knn_metric, batch_size, mlp_act,T):
        super(UserItemGraph, self).__init__()
        self.batch_size = batch_size
        self.k = k
        self.knn_metric = knn_metric
        self.mlp_act = mlp_act
        #self.dynamic_k = interaction_num
        self.T  = T
        '''
        # Define layers
        self.user_layers = nn.ModuleList([Attentive(user_isize) for _ in range(n_attlayers)] + 
                                         [nn.Linear(user_isize if i == 0 else hidden_size, hidden_size) for i in range(n_mlplayers - 1)] + 
                                         [nn.Linear(hidden_size, outsize)])
        self.item_layers = nn.ModuleList([Attentive(item_isize) for _ in range(n_attlayers)] + 
                                         [nn.Linear(item_isize if i == 0 else hidden_size, hidden_size) for i in range(n_mlplayers - 1)] + 
                                         [nn.Linear(hidden_size, outsize)])
        '''
        #self.user_Embedding = nn.Embedding.from_pretrained(user_features, freeze=False)
        self.user_layers = nn.ModuleList([Attentive(user_isize) for _ in range(n_attlayers)] + [nn.Linear(user_isize, hidden_size) ] + [nn.Linear(hidden_size, hidden_size) for i in range(n_mlplayers-2)] + [nn.Linear(hidden_size, user_isize)])
        self.item_layers = nn.ModuleList([Attentive(item_isize) for _ in range(n_attlayers)] + [nn.Linear(item_isize, hidden_size) ] + [nn.Linear(hidden_size, hidden_size) for i in range(n_mlplayers-2)] + [nn.Linear(hidden_size, item_isize)])
    def internal_forward(self, h, layers):

        for i, layer in enumerate(layers):
            h = layer(h)
            if i != (len(layers) - 1):  # Apply activation except for the last layer
                h = F.relu(h) if self.mlp_act == "relu" else F.tanh(h)
        return h
    def forward(self, user_features, item_features):
        user_embeddings = self.internal_forward(user_features, self.user_layers)
        item_embeddings = self.internal_forward(item_features, self.item_layers)
        #indices, weights = build_knn_edges(user_embeddings, item_embeddings, self.k, self.batch_size)
        rows,cols, weights = knn_fast_threshold(user_embeddings, item_embeddings, self.batch_size, self.T)
        end_time = time.time()
        row = torch.cat((rows, cols))
        col = torch.cat((cols, rows))
        #row = torch.cat((indices[0], indices[1]))
        #col = torch.cat((indices[1], indices[0]))
        weights = torch.cat((weights, weights))
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        weights = weights.detach().cpu().numpy()
        adj = sp.coo_matrix((weights, (row, col)), shape=(user_features.shape[0] + item_features.shape[0], user_features.shape[0] + item_features.shape[0]))
        #adj = convert_sp_mat_to_sp_tensor(adj_coo).coalesce()
        #g = dgl.graph((row, col), num_nodes = user_features.shape[0] + item_features.shape[0])
        #g.edata['w'] = torch.tensor(weights, dtype=torch.float32)
        #g = g.to('cuda')
        re_adj = convert_sp_mat_to_sp_tensor(adj).coalesce()
        return re_adj
      
class graph_maker(nn.Module):
    def __init__(self, n_users, m_items, item_features, n_layers, mlp_act='relu', k=20):
      super(graph_maker, self).__init__()
      self.n_users = n_users
      self.m_items = m_items
      self.item_features = item_features
      self.n_layers = n_layers
      self.mlp_act = mlp_act
      self.item_layers = nn.ModuleList([Attentive(item_features.shape[1]) for i in range(n_layers)])
      self.k_param = nn.Parameter(torch.tensor(k, dtype=torch.float32))
    
    def internal_forward(self, h, layers):
        for i, layer in enumerate(layers):
            h = layer(h)
            if i != (len(layers) - 1):  # Apply activation except for the last layer
                h = F.relu(h) if self.mlp_act == "relu" else F.tanh(h)
        return h
    
    def forward(self, graph, k, b):
        item_embeddings = self.internal_forward(self.item_features, self.item_layers)
        int_k = int(self.k_param.item())
        rows, cols, weights = add_edge(graph, item_embeddings, self.n_users, self.m_items, int_k, b)
        row = torch.cat([rows, cols])
        col = torch.cat([cols, rows])
        weights = torch.cat([weights, weights])
        # スパーステンソルの非ゼロ要素を1に変える
        indices = graph._indices()
        values = torch.ones_like(graph._values())
        adj = torch.sparse_coo_tensor(indices, values, graph.size(), device=graph.device)
        #weights = torch.ones_like(weights)
        # 追加するエッジのインデックスをgraphに追加
        new_indices = torch.stack([row, col], dim=0)
        new_edges_tensor = torch.sparse_coo_tensor(new_indices, weights, adj.size())
        graph = adj + new_edges_tensor
        
        return graph
    
class graph_maker2(nn.Module):
    def __init__(self, n_users, m_items, item_features, n_layers, T, mlp_act='relu', k=20):
      super(graph_maker2, self).__init__()
      self.n_users = n_users
      self.m_items = m_items
      self.item_features = item_features
      self.n_layers = n_layers
      self.mlp_act = mlp_act
      #self.item_embeddings = nn.Parameter(item_features)
      self.softmax = nn.Softmax(dim=0)
      self.modal_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
      self.layer = nn.ModuleList([nn.Linear(item_features.shape[1], item_features.shape[1]) for i in range(n_layers-1)] + [nn.Linear(item_features.shape[1], args.latdim)])
      #self.item_layers = nn.ModuleList([Attentive(item_features.shape[1]) for i in range(n_layers)])
      #self.T = nn.Parameter(torch.Tensor(T, dtype=torch.float32))
      #self.k_param = nn.Parameter(torch.tensor(k, dtype=torch.float32))
    
    def internal_forward(self, h, layers):
        for i, layer in enumerate(layers):
            h = layer(h)
            if i != (len(layers) - 1):  # Apply activation except for the last layer
                h = F.relu(h) if self.mlp_act == "relu" else F.tanh(h)
        return h
    
    def forward(self, graph, k, original_item_embeddings, b):
        item_embeddings_feats = self.internal_forward(self.item_features, self.layer)
        #item_embeddings = self.internal_forward(self.item_features, self.item_layers)
        #T = torch.sigmoid(self.T) * 2 - 1  # Tを-1から1の範囲に正規化
        #int_k = int(self.k_param.item())
        weights = self.softmax(self.modal_weights)
        item_embeddings = weights[0] * item_embeddings_feats + weights[1] * original_item_embeddings
        rows, cols, weights = add_edge2(graph, item_embeddings, self.n_users, self.m_items, k, b)
        row = torch.cat([rows, cols])
        col = torch.cat([cols, rows])
        weights = torch.cat([weights, weights])
        # スパーステンソルの非ゼロ要素を1に変える
        indices = graph._indices()
        values = torch.ones_like(graph._values())
        adj = torch.sparse_coo_tensor(indices, values, graph.size(), device=graph.device)
        #weights = torch.ones_like(weights)
        # 追加するエッジのインデックスをgraphに追加
        new_indices = torch.stack([row, col], dim=0)
        new_edges_tensor = torch.sparse_coo_tensor(new_indices, torch.ones(len(weights)).cuda(), adj.size())
        graph = adj + new_edges_tensor
        
        return graph

class graph_maker3(nn.Module):
    def __init__(self, n_users, m_items, item_features, n_layers, mlp_act='relu', k=20):
      super(graph_maker3, self).__init__()
      self.n_users = n_users
      self.m_items = m_items
      self.item_features = item_features
      self.n_layers = n_layers
      self.mlp_act = mlp_act
      #self.item_layers = nn.ModuleList([Attentive(item_features.shape[1]) for i in range(n_layers)])
      #self.k_param = nn.Parameter(torch.tensor(k, dtype=torch.float32))
    
    def internal_forward(self, h, layers):
        for i, layer in enumerate(layers):
            h = layer(h)
            if i != (len(layers) - 1):  # Apply activation except for the last layer
                h = F.relu(h) if self.mlp_act == "relu" else F.tanh(h)
        return h
    
    def forward(self, graph, k, b):
        item_embeddings = self.internal_forward(self.item_features, self.item_layers)
        int_k = int(self.k_param.item())
        rows, cols, weights = add_edge(graph, item_embeddings, self.n_users, self.m_items, int_k, b)
        row = torch.cat([rows, cols])
        col = torch.cat([cols, rows])
        weights = torch.cat([weights, weights])
        # スパーステンソルの非ゼロ要素を1に変える
        indices = graph._indices()
        values = torch.ones_like(graph._values())
        adj = torch.sparse_coo_tensor(indices, values, graph.size(), device=graph.device)
        #weights = torch.ones_like(weights)
        # 追加するエッジのインデックスをgraphに追加
        new_indices = torch.stack([row, col], dim=0)
        new_edges_tensor = torch.sparse_coo_tensor(new_indices, weights, adj.size())
        graph = adj + new_edges_tensor
        
        return graph
    
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # x: (N, in_features)
        # adj: (N, N) のスパース隣接行列
        x = torch.sparse.mm(adj, x)
        x = self.linear(x)
        return x

class UserItemGraph2(nn.Module):
    def __init__(self,user_features, item_features, interaction_num, n_gcn_layers, knn_metric, batch_size, mlp_act):
        super(UserItemGraph2, self).__init__()
        self.batch_size = batch_size
        #self.k = k
        self.knn_metric = knn_metric
        self.mlp_act = mlp_act
        self.dynamic_k = interaction_num
        self.user_features = user_features
        self.item_features = item_features
        self.features = torch.cat((user_features, item_features), 0)
        in_dim = user_features.shape[1]
        self.gcn = nn.ModuleList([GraphConvolution(in_dim, in_dim) for i in range(n_gcn_layers)])

        '''
        # Define layers
        self.user_layers = nn.ModuleList([Attentive(user_isize) for _ in range(n_attlayers)] + 
                                         [nn.Linear(user_isize if i == 0 else hidden_size, hidden_size) for i in range(n_mlplayers - 1)] + 
                                         [nn.Linear(hidden_size, outsize)])
        self.item_layers = nn.ModuleList([Attentive(item_isize) for _ in range(n_attlayers)] + 
                                         [nn.Linear(item_isize if i == 0 else hidden_size, hidden_size) for i in range(n_mlplayers - 1)] + 
                                         [nn.Linear(hidden_size, outsize)])
        '''
        #self.user_Embedding = nn.Embedding.from_pretrained(user_features, freeze=False)
        #self.user_layers = nn.ModuleList([Attentive(user_isize) for _ in range(n_attlayers)] + [nn.Linear(user_isize, hidden_size) ] + [nn.Linear(hidden_size, hidden_size) for i in range(n_mlplayers-2)] + [nn.Linear(hidden_size, user_isize)])
        #self.item_layers = nn.ModuleList([Attentive(item_isize) for _ in range(n_attlayers)] + [nn.Linear(item_isize, hidden_size) ] + [nn.Linear(hidden_size, hidden_size) for i in range(n_mlplayers-2)] + [nn.Linear(hidden_size, item_isize)])
    def internal_forward(self, h, adj):
        for i, gcn in enumerate(self.gcn):
            h = gcn(h, adj)
            if i != (len(self.gcn) - 1):
                h = F.relu(h) if self.mlp_act == "relu" else F.tanh(h)
        return h
    def forward(self, adj):
        embeddings = self.internal_forward(self.features, adj)
        user_embeddings = embeddings[:self.user_features.shape[0]]
        item_embeddings = embeddings[self.user_features.shape[0]:]
        #indices, weights = build_knn_edges(user_embeddings, item_embeddings, self.k, self.batch_size)
        rows,cols, weights = knn_dynamic(user_embeddings, item_embeddings, self.dynamic_k, self.batch_size)
        end_time = time.time()
        row = torch.cat((rows, cols))
        col = torch.cat((cols, rows))
        #row = torch.cat((indices[0], indices[1]))
        #col = torch.cat((indices[1], indices[0]))
        weights = torch.cat((weights, weights))
        row = row.cpu().numpy()
        col = col.cpu().numpy()
        weights = weights.detach().cpu().numpy()
        weights = np.ones(len(row))
        adj = sp.coo_matrix((weights, (row, col)), shape=(self.user_features.shape[0] + self.item_features.shape[0], self.user_features.shape[0] + self.item_features.shape[0]))
        #adj = convert_sp_mat_to_sp_tensor(adj_coo).coalesce()
        #g = dgl.graph((row, col), num_nodes = user_features.shape[0] + item_features.shape[0])
        #g.edata['w'] = torch.tensor(weights, dtype=torch.float32)
        #g = g.to('cuda')
        re_adj = convert_sp_mat_to_sp_tensor(adj).coalesce()
        return re_adj
    
class graph_learner(nn.Module):
    def __init__(self, indim, interaction_num, n_layers, knn_metric, batch_size, mlp_act):
        super(graph_learner, self).__init__()
        self.interaction_num = interaction_num
        self.knn_metric = knn_metric
        self.batch_size = batch_size
        self.mlp_act = mlp_act

        self.item_layers = nn.ModuleList([Attentive(indim) for i in range(n_layers)])
        

    def internal_forward(self, h, layers):
        for i, layer in enumerate(layers):
            h = layer(h)
            if i != (len(layers) - 1):
                h = F.relu(h) if self.mlp_act == "relu" else F.tanh(h)
        return h
    def forward(self, user_final_embeddings, item_final_embeddings, user_features, item_features,k, b):
        user_features = self.internal_forward(user_features, self.item_layers)
        item_features = self.internal_forward(item_features, self.item_layers)
