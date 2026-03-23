import torch
import torch.nn as nn
import torch.nn.functional as F


def init_orthogonal(layer):
    """Apply orthogonal initialization to linear layers."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class BipartiteGraphConvolution(nn.Module):
    """
    Generic bipartite graph convolution.

    It covers both cases used in this repo:
    - symmetric feature sizes for the standard GNN
    - asymmetric feature sizes for the DEQ variant
    """

    def __init__(self, left_dim, right_dim, right_to_left=False):
        super().__init__()

        self.left_dim = left_dim
        self.right_dim = right_dim
        self.right_to_left = right_to_left

        self.feature_module_left = nn.Sequential(
            nn.Linear(left_dim, left_dim),
            nn.ReLU(),
            nn.Linear(left_dim, left_dim),
        )
        self.feature_module_edge = nn.Sequential()
        self.feature_module_right = nn.Sequential(
            nn.Linear(right_dim, right_dim, bias=False),
            nn.ReLU(),
            nn.Linear(right_dim, right_dim),
        )

        if self.right_to_left:
            output_in_dim = right_dim + left_dim
            output_dim = left_dim
        else:
            output_in_dim = left_dim + right_dim
            output_dim = right_dim

        self.output_module = nn.Sequential(
            nn.Linear(output_in_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        self.apply(init_orthogonal)

    def forward(self, inputs):
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs

        if self.right_to_left:
            scatter_dim = 0
            prev_features = self.feature_module_left(left_features)
            joint_features = self.feature_module_edge(edge_features)
            joint_features = joint_features * self.feature_module_right(right_features)[edge_indices[1]]
        else:
            scatter_dim = 1
            prev_features = self.feature_module_right(right_features)
            joint_features = self.feature_module_edge(edge_features)
            joint_features = joint_features * self.feature_module_left(left_features)[edge_indices[0]]

        conv_output = torch.zeros(
            scatter_out_size,
            joint_features.shape[1],
            device=left_features.device,
        )
        conv_output.index_add_(0, edge_indices[scatter_dim], joint_features)

        return self.output_module(torch.cat([conv_output, prev_features], dim=1))


class GNN(nn.Module):
    """Standard four-layer GNN policy."""

    def __init__(self, emb_size, n_cons_feats, n_edge_feats, n_var_feats):
        super().__init__()

        self.emb_size = emb_size
        self.cons_nfeats = n_cons_feats
        self.edge_nfeats = n_edge_feats
        self.var_nfeats = n_var_feats

        self.cons_embedding = nn.Sequential(
            nn.Linear(n_cons_feats, emb_size),
            nn.ReLU(),
        )
        self.var_embedding = nn.Sequential(
            nn.Linear(n_var_feats, emb_size),
            nn.ReLU(),
        )
        self.edge_embedding = nn.Sequential()

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size, emb_size, right_to_left=True)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size, emb_size)
        self.conv_v_to_c2 = BipartiteGraphConvolution(emb_size, emb_size, right_to_left=True)
        self.conv_c_to_v2 = BipartiteGraphConvolution(emb_size, emb_size)

        self.output_module = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1, bias=False),
        )

        self.apply(init_orthogonal)

    def forward(self, inputs):
        constraint_features, edge_indices, edge_features, variable_features, n_cons_total, n_vars_total = inputs

        constraint_features = self.cons_embedding(constraint_features)
        variable_features = self.var_embedding(variable_features)
        edge_features = self.edge_embedding(edge_features)

        constraint_features = F.relu(
            self.conv_v_to_c((constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
        )
        variable_features = F.relu(
            self.conv_c_to_v((constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        )
        constraint_features = F.relu(
            self.conv_v_to_c2((constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
        )
        variable_features = F.relu(
            self.conv_c_to_v2((constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        )

        return self.output_module(variable_features)


class DEQGNNLayer(nn.Module):
    """The fixed-point update function used by DEQGNN."""

    def __init__(self, emb_size):
        super().__init__()
        var_feature_size = emb_size + 1

        self.conv_v_to_c = BipartiteGraphConvolution(emb_size, var_feature_size, right_to_left=True)
        self.conv_c_to_v = BipartiteGraphConvolution(emb_size, var_feature_size)
        self.conv_v_to_c2 = BipartiteGraphConvolution(emb_size, var_feature_size, right_to_left=True)
        self.conv_c_to_v2 = BipartiteGraphConvolution(emb_size, var_feature_size)
        self.to_y = nn.Linear(var_feature_size, 1)

    def forward(self, y, x):
        static_var_feats, static_cons_feats, edge_indices, edge_features, n_cons_total, n_vars_total = x

        variable_features = torch.cat([static_var_feats, y], dim=1)
        constraint_features = F.relu(
            self.conv_v_to_c((static_cons_feats, edge_indices, edge_features, variable_features, n_cons_total))
        )
        variable_features = F.relu(
            self.conv_c_to_v((constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        )
        constraint_features = F.relu(
            self.conv_v_to_c2((constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
        )
        variable_features = F.relu(
            self.conv_c_to_v2((constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        )

        return self.to_y(variable_features)


class DEQGNN(nn.Module):
    """DEQ-style GNN implemented with a fixed number of native PyTorch iterations."""

    def __init__(self, emb_size, n_cons_feats, n_edge_feats, n_var_feats, num_iters=10):
        super().__init__()

        self.num_iters = num_iters
        self.cons_embedding = nn.Sequential(
            nn.Linear(n_cons_feats, emb_size),
            nn.ReLU(),
        )
        self.var_embedding = nn.Sequential(
            nn.Linear(n_var_feats, emb_size),
            nn.ReLU(),
        )
        self.deq_func = DEQGNNLayer(emb_size)
        self.output_module = nn.Sequential(
            nn.Linear(emb_size + 1, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1, bias=False),
        )

        self.apply(init_orthogonal)

    def forward(self, inputs):
        constraint_features, edge_indices, edge_features, variable_features, n_cons_total, n_vars_total = inputs

        static_cons_feats = self.cons_embedding(constraint_features)
        static_var_feats = self.var_embedding(variable_features)
        y = torch.zeros(n_vars_total, 1, device=variable_features.device)
        static_x = (static_var_feats, static_cons_feats, edge_indices, edge_features, n_cons_total, n_vars_total)

        if not self.training:
            with torch.no_grad():
                for _ in range(self.num_iters):
                    y = self.deq_func(y, static_x)
        else:
            for _ in range(self.num_iters):
                y = self.deq_func(y, static_x)

        final_features = torch.cat([static_var_feats, y], dim=1)
        return self.output_module(final_features)
