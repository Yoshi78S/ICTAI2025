import torch
import torch.nn as nn

class Attentive(nn.Module):
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        return x @ torch.diag(self.w)
      
class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
        rc = x._indices()[:,mask]
        val = x._values()[mask]*(1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)

class GCNConv_Sparse(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_Sparse, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, adj):
        # x: Node features, shape: (num_nodes, input_size)
        # adj: Sparse adjacency matrix, shape: (num_nodes, num_nodes)

        # Apply the linear transformation
        x_transformed = self.linear(x)  # shape: (num_nodes, output_size)
        
        # Perform sparse-dense matrix multiplication
        if isinstance(adj, torch.Tensor) and adj.is_sparse:
            out = torch.sparse.mm(adj, x_transformed)  # adjacency matrix as sparse
        else:
            raise ValueError("Input adjacency matrix must be a sparse tensor")

        # Return the output features
        return out