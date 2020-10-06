from __future__ import print_function
import os
import ops
import sys
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('%s/pytorch_structure2vec-master/s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from s2v_lib import S2VLIB # noqa
from pytorch_util import weights_init, gnn_spmm # noqa


class GXN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats,
                 latent_dim=[32, 32, 32, 1], k=30, ks=[0.9,0.7],
                 conv1d_channels=[16, 32],
                 conv1d_kws=[0, 5],
                 cross_weight=1.0, fuse_weight=1.0, R=1):
        print('Initializing GXN')
        super(GXN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i-1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)

        dense_dim = int((k-2)/2+1)
        self.dense_dim = (dense_dim-conv1d_kws[1]+1)*conv1d_channels[1]

        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

        self.ks = ks
        self.gxn = ops.GraphCrossnet(ks, num_node_feats, 97, cross_weight=cross_weight, fuse_weight=fuse_weight, R=R)

        weights_init(self)

    def forward(self, graph_list, node_feat, edge_feat):

        device = torch.device(node_feat.device)

        graph_sizes = [graph_list[i].num_nodes for i in range(len(graph_list))]
        node_degs = [torch.Tensor(graph_list[i].degs)+1 for i in range(len(graph_list))]
        node_degs = torch.cat(node_degs).unsqueeze(1)

        n2n_sp, e2n_sp, subg_sp = S2VLIB.PrepareMeanField(graph_list)
        n2n_sp = n2n_sp.to(device)
        e2n_sp = e2n_sp.to(device)
        subg_sp = subg_sp.to(device)
        node_degs = node_degs.to(device)

        node_feat = Variable(node_feat)

        if edge_feat is not None:
            edge_feat = Variable(edge_feat)
        n2n_sp = Variable(n2n_sp)
        e2n_sp = Variable(e2n_sp)
        subg_sp = Variable(subg_sp)
        node_degs = Variable(node_degs)

        h = self.sortpooling_embedding(node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs)
        return h

    def sortpooling_embedding(self, node_feat, edge_feat, n2n_sp, e2n_sp, subg_sp, graph_sizes, node_degs):

        device = torch.device(node_feat.device)

        ''' if exists edge feature, concatenate to node feature vector '''
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = gnn_spmm(e2n_sp, input_edge_linear)
            node_feat = torch.cat([node_feat, e2npool_input], 1)

        ''' graph convolution layers '''
        A = ops.spec_normalize_adj(n2n_sp).to(device)

        ver = 2
        if ver == 2:
            cur_message_layer, ret_s1, ret_s2 = self.gxn(A, node_feat)
        else:
            lv = 0
            cur_message_layer = node_feat
            cat_message_layers = []
            while lv < len(self.latent_dim):
                n2npool = gnn_spmm(n2n_sp, cur_message_layer) + cur_message_layer   # noqa
                node_linear = self.conv_params[lv](n2npool)                         # Y = Y * W
                normalized_linear = node_linear.div(node_degs)                      # Y = D^-1 * Y
                cur_message_layer = F.tanh(normalized_linear)
                cat_message_layers.append(cur_message_layer)
                lv += 1

            cur_message_layer = torch.cat(cat_message_layers, 1)

        ''' sortpooling layer '''
        sort_channel = cur_message_layer[:, -1]
        batch_sortpooling_graphs = torch.zeros(len(graph_sizes), self.k, self.total_latent_dim).to(device)
        batch_sortpooling_graphs = Variable(batch_sortpooling_graphs)
        
        accum_count = 0
        for i in range(subg_sp.size()[0]):
            to_sort = sort_channel[accum_count: accum_count + graph_sizes[i]]
            k = self.k if self.k <= graph_sizes[i] else graph_sizes[i]
            _, topk_indices = to_sort.topk(k)
            topk_indices += accum_count
            sortpooling_graph = cur_message_layer.index_select(0, topk_indices)
            if k < self.k:
                to_pad = torch.zeros(self.k-k, self.total_latent_dim).to(device)
                to_pad = Variable(to_pad)
                sortpooling_graph = torch.cat((sortpooling_graph, to_pad), 0)
            batch_sortpooling_graphs[i] = sortpooling_graph
            accum_count += graph_sizes[i]

        ''' traditional 1d convlution and dense layers '''
        to_conv1d = batch_sortpooling_graphs.view((-1, 1, self.k * self.total_latent_dim))

        conv1d_res = self.conv1d_params1(to_conv1d)
        conv1d_res = F.relu(conv1d_res)
        conv1d_res = self.maxpool1d(conv1d_res)
        conv1d_res = self.conv1d_params2(conv1d_res)
        conv1d_res = F.relu(conv1d_res)

        to_dense = conv1d_res.view(len(graph_sizes), -1)

        if self.output_dim > 0:
            out_linear = self.out_params(to_dense)
            reluact_fp = F.relu(out_linear)
        else:
            reluact_fp = to_dense

        return F.relu(reluact_fp), ret_s1, ret_s2