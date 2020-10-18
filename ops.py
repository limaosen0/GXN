import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np
import scipy.sparse as sp


def spec_normalize_adj(adj, high_order=False):
    adj = adj.to_dense().cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return torch.FloatTensor(adj_norm.todense())


def spac_normalize_adj(adj, high_order=False):
    adj = adj.to_dense().cpu().numpy()
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1.).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm = adj.dot(d_mat_inv_sqrt).transpose().tocoo()
    return torch.FloatTensor(adj_norm.todense())


def normalize_adj_torch(mx):
    mx = mx.to_dense()
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=bias)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x_fts = self.fc(x)
        if self.bias is not None:
            x_fts += self.bias
        return self.act(x_fts)


class GCN_MI(nn.Module):
    def __init__(self, in_ft, out_ft, act='prelu', bias=True):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, A, x, sparse=False):
        x_fts = self.fc(x)
        if sparse:
            out = torch.unsqueeze(torch.spmm(A, torch.squeeze(x_fts, 0)), 0)
        else:
            out = torch.bmm(A.unsqueeze(0), x_fts.unsqueeze(0))
        if self.bias is not None:
            out += self.bias
        return self.act(out).squeeze(0)


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, A, X, act=None):
        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        if act is not None:
            X = act(X)
        return X


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super().__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), -2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), -2)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 0).squeeze(-1)
        v = logits.shape[0]

        return logits, logits[:v//2]


class GraphCrossnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim=48, cross_weight=1.0, fuse_weight=1.0, R=1, cross_layer=2):
        super(GraphCrossnet, self).__init__()
        self.ks = ks
        self.cs_w = cross_weight
        self.fs_w = fuse_weight
        self.cs_l = cross_layer

        self.start_gcn_s1 = GCN(in_dim, dim)
        self.start_gcn_s2 = GCN(dim, dim)
        self.end_gcn = GCN(2*dim, out_dim)

        self.index_select_s1 = IndexSelect(ks[0], dim, act='prelu', R=R)
        self.index_select_s2 = IndexSelect(ks[1], dim, act='prelu', R=R)
        self.pool_s12_start = GraphPool(dim)
        self.pool_s23_start = GraphPool(dim)
        self.unpool_s21_end = GraphUnpool(dim)
        self.unpool_s32_end = GraphUnpool(dim)

        self.s1_l1 = GCN(dim, dim)
        self.s1_l2 = GCN(dim, dim)
        self.s1_l3 = GCN(dim, dim)
        self.s2_l1 = GCN(dim, dim)
        self.s2_l2 = GCN(dim, dim)
        self.s2_l3 = GCN(dim, dim)
        self.s3_l1 = GCN(dim, dim)
        self.s3_l2 = GCN(dim, dim)
        self.s3_l3 = GCN(dim, dim)

        if self.cs_l>=1:
            self.pool_s12_1 = GraphPool(dim, g=True)
            self.unpool_s21_1 = GraphUnpool(dim)
            self.pool_s23_1 = GraphPool(dim, g=True)
            self.unpool_s32_1 = GraphUnpool(dim)
        if self.cs_l>=2:
            self.pool_s12_2 = GraphPool(dim, g=True)
            self.unpool_s21_2 = GraphUnpool(dim)
            self.pool_s23_2 = GraphPool(dim, g=True)
            self.unpool_s32_2 = GraphUnpool(dim)

    def forward(self, A, x):

        A_s1  = A
        x_s1  = self.start_gcn_s1(A_s1, x)
        x_org = x_s1
        x_s1_ = torch.zeros_like(x_s1)
        x_s1_ = x_s1[torch.randperm(x_s1.shape[0]),:]
        ret_s1, value_s1, idx_s1, idx_s1_, Xdown_s1 = self.index_select_s1(x_s1, x_s1_, A_s1)        
        x_s2, A_s2 = self.pool_s12_start(A_s1, x_s1, idx_s1, idx_s1_, value_s1, initlayer=True)

        x_s2 = self.start_gcn_s2(A_s2, x_s2)
        x_s2_ = torch.zeros_like(x_s2)
        x_s2_ = x_s2[torch.randperm(x_s2.shape[0]),:]
        ret_s2, value_s2, idx_s2, idx_s2_, Xdown_s2 = self.index_select_s2(x_s2, x_s2_, A_s2)
        x_s3, A_s3 = self.pool_s23_start(A_s2, x_s2, idx_s2, idx_s2_, value_s2, initlayer=True)

        res_s1_0, res_s2_0, res_s3_0 = x_s1, x_s2, x_s3

        x_s1 = self.s1_l1(A_s1, x_s1, F.relu)
        x_s2 = self.s2_l1(A_s2, x_s2, F.relu)
        x_s3 = self.s3_l1(A_s3, x_s3, F.relu)

        res_s1_1, res_s2_1, res_s3_1 = x_s1, x_s2, x_s3

        if self.cs_l >= 1:
            x_s12_fu = self.pool_s12_1(A_s1, x_s1, idx_s1, idx_s1_, value_s1)
            x_s21_fu = self.unpool_s21_1(A_s1, x_s2, idx_s1)
            x_s23_fu = self.pool_s23_1(A_s2, x_s2, idx_s2, idx_s2_, value_s2)
            x_s32_fu = self.unpool_s32_1(A_s2, x_s3, idx_s2)

            x_s1 = x_s1 + self.cs_w * x_s21_fu + res_s1_0
            x_s2 = x_s2 + self.cs_w * (x_s12_fu + x_s32_fu)/2 + res_s2_0
            x_s3 = x_s3 + self.cs_w * x_s23_fu + res_s3_0

        x_s1 = self.s1_l2(A_s1, x_s1, F.relu)
        x_s2 = self.s2_l2(A_s2, x_s2, F.relu)
        x_s3 = self.s3_l2(A_s3, x_s3, F.relu)

        if self.cs_l >= 2:
            x_s12_fu = self.pool_s12_2(A_s1, x_s1, idx_s1, idx_s1_, value_s1)
            x_s21_fu = self.unpool_s21_2(A_s1, x_s2, idx_s1)
            x_s23_fu = self.pool_s23_2(A_s2, x_s2, idx_s2, idx_s2_, value_s2)
            x_s32_fu = self.unpool_s32_2(A_s2, x_s3, idx_s2)

            x_s1 = x_s1 + self.cs_w * 0.05 * x_s21_fu
            x_s2 = x_s2 + self.cs_w * 0.05 * (x_s12_fu + x_s32_fu)/2
            x_s3 = x_s3 + self.cs_w * 0.05 * x_s23_fu

        x_s1 = self.s1_l3(A_s1, x_s1, F.relu)
        x_s2 = self.s2_l3(A_s2, x_s2, F.relu)
        x_s3 = self.s3_l3(A_s3, x_s3, F.relu)
        
        x_s3_out = self.unpool_s32_end(A_s2, x_s3, idx_s2) + Xdown_s2
        x_s2_out = self.unpool_s21_end(A_s1, x_s2 + x_s3_out, idx_s1)
        x_agg = x_s1 + x_s2_out * self.fs_w + Xdown_s1 * self.fs_w
        x_agg = torch.cat([x_agg, x_org], 1)
        x_agg = self.end_gcn(A_s1, x_agg)

        return x_agg, ret_s1, ret_s2


class IndexSelect(nn.Module):

    def __init__(self, k, n_h, act,  R=1):
        super().__init__()
        self.k = k
        self.R = R
        self.sigm = nn.Sigmoid()
        self.fc = MLP(n_h, n_h, act)
        self.disc = Discriminator(n_h)
        self.gcn1 = GCN(n_h, n_h)

    def forward(self, seq1, seq2, A, samp_bias1=None, samp_bias2=None):
        h_1 = self.fc(seq1)
        h_2 = self.fc(seq2)
        h_n1 = self.gcn1(A, h_1)

        X = self.sigm(h_n1)
        ret, ret_true = self.disc(X, h_1, h_2, samp_bias1, samp_bias2)
        scores = self.sigm(ret_true).squeeze()
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(num_nodes))
        values1, idx1 = values[:int(self.k*num_nodes)], idx[:int(self.k*num_nodes)]
        values0, idx0 = values[int(self.k*num_nodes):], idx[int(self.k*num_nodes):]

        return ret, values1, idx1, idx0, h_n1


class GraphPool(nn.Module):

    def __init__(self, in_dim, g=False):
        super(GraphPool, self).__init__()
        self.g = g
        if self.g:
            self.down_gcn = GCN(in_dim, in_dim)
        
    def forward(self, A, X, idx, idx_=None, value=None, initlayer=False):
        if self.g:
            X = self.down_gcn(A, X)

        new_x = X[idx,:]
        score = torch.unsqueeze(value, -1)
        new_x = torch.mul(new_x, score)

        if initlayer:
            A = self.removeedge(A, idx)
            return new_x, A
        else:
            return new_x

    def removeedge(self, A, idx):
        A_ = A[idx,:]
        A_ = A_[:,idx]
        return A_

    

class GraphUnpool(nn.Module):

    def __init__(self, in_dim):
        super(GraphUnpool, self).__init__()
        self.up_gcn = GCN(in_dim, in_dim)

    def forward(self, A, X, idx):

        new_X = torch.zeros([A.shape[0], X.shape[1]]).to(X.device)
        new_X[idx] = X
        new_X = self.up_gcn(A, new_X)
        return new_X