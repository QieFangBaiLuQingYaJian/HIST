import torch
import numpy as np
import torch.nn.functional as F
import torch.nn.parameter as Parameter
from utils.utils_gcn import get_param, ccorr, rotate, softmax
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import MessagePassing


class StarEConvLayer(MessagePassing):
    """ The important stuff. """

    def __init__(self, in_channels, out_channels, num_rels, act=lambda x: x,
                 config=None):
        super(self.__class__, self).__init__(flow='target_to_source',
                                             aggr='add')

        self.p = config
        self.in_channels = in_channels #200
        self.out_channels = out_channels #200
        self.num_rels = num_rels #532
        self.act = act
        self.device = None

        self.w_loop = get_param((in_channels, out_channels))  # (100,200)
        self.w_in = get_param((in_channels, out_channels))  # (100,200)
        self.w_out = get_param((in_channels, out_channels))  # (100,200)
        self.w_rel = get_param((in_channels, out_channels))  # (100,200)

        if self.p['STATEMENT_LEN'] != 3:#-1
            if self.p['STAREARGS']['QUAL_AGGREGATE'] == 'sum' or self.p['STAREARGS']['QUAL_AGGREGATE'] == 'mul':#sum
                self.w_q = get_param((in_channels, in_channels))  # new for quals setup
            elif self.p['STAREARGS']['QUAL_AGGREGATE'] == 'concat':
                self.w_q = get_param((2 * in_channels, in_channels))  # need 2x size due to the concat operation

        self.loop_rel = get_param((1, in_channels))  # (1,100)
        self.loop_ent = get_param((1, in_channels))  # new

        self.drop = torch.nn.Dropout(self.p['STAREARGS']['GCN_DROP'])
        self.bn = torch.nn.BatchNorm1d(out_channels)

        if self.p['STAREARGS']['ATTENTION']:
            assert self.p['STAREARGS']['GCN_DIM'] == self.p['EMBEDDING_DIM'], "Current attn implementation requires those tto be identical"
            assert self.p['EMBEDDING_DIM'] % self.p['STAREARGS']['ATTENTION_HEADS'] == 0, "should be divisible"
            self.heads = self.p['STAREARGS']['ATTENTION_HEADS']
            self.attn_dim = self.out_channels // self.heads
            self.negative_slope = self.p['STAREARGS']['ATTENTION_SLOPE']
            self.attn_drop = self.p['STAREARGS']['ATTENTION_DROP']
            self.att = get_param((1, self.heads, 2 * self.attn_dim))

        if self.p['STAREARGS']['BIAS']: self.register_parameter('bias', Parameter(
            torch.zeros(out_channels)))

    def forward(self, x, edge_index, edge_type, rel_embed,
                qualifier_ent=None, qualifier_rel=None, quals=None):
        # x(47156,200) ;edge_index(2,380696); edge_type(380696,) ; rel_embed(1064,200) ; qualifier_ent:None ;qualifier_rel:None ; quals: (3,74866)
        """

        See end of doc string for explaining.

        :param x: all entities*dim_of_entities (for jf17k -> 28646*200)
        :param edge_index: COO matrix (2 list each having nodes with index
        [1,2,3,4,5]
        [3,4,2,5,4]

        Here node 1 and node 3 are connected with edge.
        And the type of edge can be found using edge_type.

        Note that there are twice the number of edges as each edge is also reversed.
        )
        :param edge_type: The type of edge connecting the COO matrix
        :param rel_embed: 2 Times Total relation * emb_dim (200 in our case and 2 Times because of inverse relations)
        :param qualifier_ent:
        :param qualifier_rel:
        :param quals: Another sparse matrix

        where
            quals[0] --> qualifier relations type
            quals[1] --> qualifier entity
            quals[2] --> index of the original COO matrix that states for which edge this qualifier exists ()


        For argument sake if a knowledge graph has following statements

        [e1,p1,e4,qr1,qe1,qr2,qe2]
        [e1,p1,e2,qr1,qe1,qr2,qe3]
        [e1,p2,e3,qr3,qe3,qr2,qe2]
        [e1,p2,e5,qr1,qe1]
        [e2,p1,e4]
        [e4,p3,e3,qr4,qe1,qr2,qe4]
        [e1,p1,e5]
                                                 (incoming)         (outgoing)
                                            <----(regular)------><---(inverse)------->
        Edge index would be             :   [e1,e1,e1,e1,e2,e4,e1,e4,e2,e3,e5,e4,e3,e5]
                                            [e4,e2,e3,e5,e4,e3,e5,e1,e1,e1,e1,e2,e4,e1]

        Edge Type would be              :   [p1,p1,p2,p2,p1,p3,p1,p1_inv,p1_inv,p2_inv,p2_inv,p1_inv,p3_inv,p1_inv]

                                            <-------on incoming-----------------><---------on outgoing-------------->
        quals would be                  :   [qr1,qr2,qr1,qr2,qr3,qr2,qr1,qr4,qr2,qr1,qr2,qr1,qr2,qr3,qr2,qr1,qr4,qr2]
                                            [qe1,qe2,qe1,qe3,qe3,qe2,qe1,qe1,qe4,qe1,qe2,qe1,qe3,qe3,qe2,qe1,qe1,qe4]
                                            [0,0,1,1,2,2,3,5,5,0,0,1,1,2,2,3,5,5]
                                            <--on incoming---><--outgoing------->

        Note that qr1,qr2... and qe1, qe2, ... all belong to the same space
        :return:
        """
        if self.device is None:
            self.device = edge_index.device

        rel_embed = torch.cat([rel_embed, self.loop_rel], dim=0)#(1064,200) => (1065,200) 第一行全是0
        num_edges = edge_index.size(1) // 2 #190348
        num_ent = x.size(0) #47156

        self.in_index, self.out_index = edge_index[:, :num_edges], edge_index[:, num_edges:] #in_dex:(2,190348) out_index:(2,190348)
        self.in_type, self.out_type = edge_type[:num_edges], edge_type[num_edges:] #in_type:(190348,) out_type:(190348,)
        # quals:Tensor(3,74866)
        if self.p['STATEMENT_LEN'] != 3: #-1
            num_quals = quals.size(1) // 2 #num_quals:37433
            self.in_index_qual_ent, self.out_index_qual_ent = quals[1, :num_quals], quals[1, num_quals:]# (37433,):tensor([ 1339, 17433, 10068,  ..., 12219, 42900, 16382]) ; (37433，) 内容一样
            self.in_index_qual_rel, self.out_index_qual_rel = quals[0, :num_quals], quals[0, num_quals:] # (37433,):tensor([362, 362, 239,  ..., 125, 125, 493]) ; (37433,) 内容一样
            self.quals_index_in, self.quals_index_out = quals[2, :num_quals], quals[2, num_quals:] # (37433,):tensor([     3,      3,      6,  ..., 190310, 190324, 190342]) ; (37433,) 内容一样

        '''
            Adding self loop by creating a COO matrix. Thus \
             loop index [1,2,3,4,5]
                        [1,2,3,4,5]
             loop type [10,10,10,10,10] --> assuming there are 9 relations


        '''
        # Self edges between all the nodes
        self.loop_index = torch.stack([torch.arange(num_ent), torch.arange(num_ent)]).to(self.device) # (2,47156):[0,1,2,...,47155]
        self.loop_type = torch.full((num_ent,), rel_embed.size(0) - 1, # tensor(47156,):([1064, 1064, 1064,  ..., 1064, 1064, 1064])
                                    dtype=torch.long).to(self.device)  # if rel meb is 500, the index of the self emb is
        # 499 .. which is just added here

        self.in_norm = self.compute_norm(self.in_index, num_ent) #(190348,):tensor([0.0000, 0.0000, 0.0000,  ..., 0.0471, 0.0000, 0.0234])
        self.out_norm = self.compute_norm(self.out_index, num_ent)#(190348,):tensor([0.0000, 0.0000, 0.7071,  ..., 0.0000, 0.0445, 0.0000])


        if self.p['STATEMENT_LEN'] != 3:
            # (47156,200)
            in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                    rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                    ent_embed=x, qualifier_ent=self.in_index_qual_ent,
                                    qualifier_rel=self.in_index_qual_rel,
                                    qual_index=self.quals_index_in,
                                    source_index=self.in_index[0])
            #(47156,200)
            loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type,
                                      rel_embed=rel_embed, edge_norm=None, mode='loop',
                                      ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                      qual_index=None, source_index=None)
            #(47156,200)
            out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                     rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                     ent_embed=x, qualifier_ent=self.out_index_qual_ent,
                                     qualifier_rel=self.out_index_qual_rel,
                                     qual_index=self.quals_index_out,
                                     source_index=self.out_index[0])

        else:
            in_res = self.propagate(self.in_index, x=x, edge_type=self.in_type,
                                    rel_embed=rel_embed, edge_norm=self.in_norm, mode='in',
                                    ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                    qual_index=None, source_index=None)

            loop_res = self.propagate(self.loop_index, x=x, edge_type=self.loop_type,
                                      rel_embed=rel_embed, edge_norm=None, mode='loop',
                                      ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                      qual_index=None, source_index=None)

            out_res = self.propagate(self.out_index, x=x, edge_type=self.out_type,
                                     rel_embed=rel_embed, edge_norm=self.out_norm, mode='out',
                                     ent_embed=None, qualifier_ent=None, qualifier_rel=None,
                                     qual_index=None, source_index=None)

        #(47156,200)
        out = self.drop(in_res) * (1 / 3) + self.drop(out_res) * (1 / 3) + loop_res * (1 / 3)

        if self.p['STAREARGS']['BIAS']:
            out = out + self.bias
        out = self.bn(out)#(47156,200)

        # Ignoring the self loop inserted, return.
        return self.act(out), torch.matmul(rel_embed, self.w_rel)[:-1]

    def rel_transform(self, ent_embed, rel_embed):
        if self.p['STAREARGS']['OPN'] == 'corr': # ent_embed:(190348,200), rel_embed:(190348,200)
            trans_embed = ccorr(ent_embed, rel_embed)
        elif self.p['STAREARGS']['OPN'] == 'sub':
            trans_embed = ent_embed - rel_embed
        elif self.p['STAREARGS']['OPN'] == 'mult':
            trans_embed = ent_embed * rel_embed
        elif self.p['STAREARGS']['OPN'] == 'rotate':
            trans_embed = rotate(ent_embed, rel_embed)
        else:
            raise NotImplementedError

        return trans_embed #(190348,200)

    def qual_transform(self, qualifier_ent, qualifier_rel):
        """
        qualifier_ent:(37433,200) ; qualifier_rel:(37433,200)
        :return:
        """
        if self.p['STAREARGS']['QUAL_OPN'] == 'corr':
            trans_embed = ccorr(qualifier_ent, qualifier_rel)
        elif self.p['STAREARGS']['QUAL_OPN'] == 'sub':
            trans_embed = qualifier_ent - qualifier_rel
        elif self.p['STAREARGS']['QUAL_OPN'] == 'mult':
            trans_embed = qualifier_ent * qualifier_rel
        elif self.p['STAREARGS']['QUAL_OPN'] == 'rotate':
            trans_embed = rotate(qualifier_ent, qualifier_rel) #(37433,200)
        else:
            raise NotImplementedError

        return trans_embed

    def qualifier_aggregate(self, qualifier_emb, rel_part_emb, alpha=0.5, qual_index=None):
        """
            In qualifier_aggregate method following steps are performed
                        qualifier_emb:(37433,200) ; rel_part_emb:(190348,200) ; qual_index:(37433):tensor([     3,      3,      6,  ..., 190310, 190324, 190342])
            qualifier_emb looks like -
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            rel_part_emb       :   [qq,ww,ee,rr,tt, .....]                      (here qq, ww, ee .. are of 200 dim)

            Note that rel_part_emb for jf17k would be around 61k*200

            Step1 : Pass the qualifier_emb to self.coalesce_quals and multiply the returned output with a weight.
            qualifier_emb   : [aa,bb,cc,dd,ee, ...... ]                 (here aa, bb, cc are of 200 dim each)
            Note that now qualifier_emb has the same shape as rel_part_emb around 61k*200

            Step2 : Combine the updated qualifier_emb (see Step1) with rel_part_emb based on defined aggregation strategy.



            Aggregates the qualifier matrix (3, edge_index, emb_dim)
        :param qualifier_emb:
        :param rel_part_emb:
        :param type:
        :param alpha
        :return:

        self.coalesce_quals    returns   :  [q+a+b+d,w+c+e+g,e'+f,......]        (here each element in the list is of 200 dim)

        """

        if self.p['STAREARGS']['QUAL_AGGREGATE'] == 'sum':
            qualifier_emb = torch.einsum('ij,jk -> ik', #矩阵乘法  (190348,200)
                                         self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0]),
                                         self.w_q)
            return alpha * rel_part_emb + (1 - alpha) * qualifier_emb      # [N_EDGES / 2 x EMB_DIM] (190348,200)
        elif self.p['STAREARGS']['QUAL_AGGREGATE'] == 'concat':
            qualifier_emb = self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0])
            agg_rel = torch.cat((rel_part_emb, qualifier_emb), dim=1)  # [N_EDGES / 2 x 2 * EMB_DIM]
            return torch.mm(agg_rel, self.w_q)                         # [N_EDGES / 2 x EMB_DIM]

        elif self.p['STAREARGS']['QUAL_AGGREGATE'] == 'mul':
            qualifier_emb = torch.mm(self.coalesce_quals(qualifier_emb, qual_index, rel_part_emb.shape[0], fill=1), self.w_q)
            return rel_part_emb * qualifier_emb
        else:
            raise NotImplementedError

    def update_rel_emb_with_qualifier(self, ent_embed, rel_embed,
                                      qualifier_ent, qualifier_rel, edge_type, qual_index=None):
        """
        The update_rel_emb_with_qualifier method performs following functions:
                ent_embed:Parameter(47156,200); rel_embed:(1065,200) ; qualifier_ent:(37433,):tensor([ 1339, 17433, 10068,  ..., 12219, 42900, 16382]) ; qualifier_rel:(37433,):tensor([362, 362, 239,  ..., 125, 125, 493]) ; edge_type:(190348):tensor([163, 270, 471,  ..., 116,  20, 503]) ; qual_index(37433):tensor([     3,      3,      6,  ..., 190310, 190324, 190342])
        Input is the secondary COO matrix (QE (qualifier entity), QR (qualifier relation), edge index (Connection to the primary COO))

        Step1 : Embed all the input
            Step1a : Embed the qualifier entity via ent_embed (So QE shape is 33k,1 -> 33k,200)
            Step1b : Embed the qualifier relation via rel_embed (So QR shape is 33k,1 -> 33k,200)
            Step1c : Embed the main statement edge_type via rel_embed (So edge_type shape is 61k,1 -> 61k,200)

        Step2 : Combine qualifier entity emb and qualifier relation emb to create qualifier emb (See self.qual_transform).
            This is generally just summing up. But can be more any pair-wise function that returns one vector for a (qe,qr) vector

        Step3 : Update the edge_type embedding with qualifier information. This uses scatter_add/scatter_mean.


        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type          :   [q,w,e',r,t,y,u,i,o,p, .....]        (here q,w,e' .. are of 200 dim each)

        After:
            edge_type          :   [q+(a+b+d),w+(c+e+g),e'+f,......]        (here each element in the list is of 200 dim)


        :param ent_embed: essentially x (28k*200 in case of Jf17k)
        :param rel_embed: essentially relation embedding matrix

        For secondary COO matrix (QE, QR, edge index)
        :param qualifier_ent:  QE
        :param qualifier_rel: QR
        edge_type:
        :return:

        index select from embedding
        phi operation between qual_ent, qual_rel
        """

        # Step 1: embedding
        qualifier_emb_rel = rel_embed[qualifier_rel] #(37433,200)
        qualifier_emb_ent = ent_embed[qualifier_ent] #(37433,200)

        rel_part_emb = rel_embed[edge_type] #(190348,200)

        # Step 2: pass it through qual_transform
        qualifier_emb = self.qual_transform(qualifier_ent=qualifier_emb_ent,# (37433,200)
                                            qualifier_rel=qualifier_emb_rel)

        # Pass it through a aggregate layer
        return self.qualifier_aggregate(qualifier_emb, rel_part_emb, alpha=self.p['STAREARGS']['TRIPLE_QUAL_WEIGHT'],
                                        qual_index=qual_index) #(190348,200)

    # return qualifier_emb
    def message(self, x_j, x_i, edge_type, rel_embed, edge_norm, mode, ent_embed=None, qualifier_ent=None,
                qualifier_rel=None, qual_index=None, source_index=None):
        #
        # x_i:(190348,200); x_j:(190348,200);
        # edge_type:(190348):tensor([163, 270, 471,  ..., 116,  20, 503])
        # rel_embed:(1065,200); ent_embed:(47156,200)
        # edge_norm:(190348):tensor([0.0000, 0.0000, 0.0000,  ..., 0.0471, 0.0000, 0.0234])
        # qualifier_index:(37433,):tensor([     3,      3,      6,  ..., 190310, 190324, 190342])
        # qualifier_ent:(37433,):tensor([ 1339, 17433, 10068,  ..., 12219, 42900, 16382])
        # qualifier_rel:(37433,):tensor([362, 362, 239,  ..., 125, 125, 493]) ; source_index:(190348,):tensor([32348, 44221, 15276,  ...,  1642, 23125, 14522])
        """

        The message method performs following functions

        Step1 : get updated relation representation (rel_embed) [edge_type] by aggregating qualifier information (self.update_rel_emb_with_qualifier).
        Step2 : Obtain edge message by transforming the node embedding with updated relation embedding (self.rel_transform).
        Step3 : Multiply edge embeddings (transform) by weight
        Step4 : Return the messages. They will be sent to subjects (1st line in the edge index COO)
        Over here the node embedding [the first list in COO matrix] is representing the message which will be sent on each edge


        More information about updating relation representation please refer to self.update_rel_emb_with_qualifier

        :param x_j: objects of the statements (2nd line in the COO)
        :param x_i: subjects of the statements (1st line in the COO)
        :param edge_type: relation types
        :param rel_embed: embedding matrix of all relations
        :param edge_norm:
        :param mode: in (direct) / out (inverse) / loop
        :param ent_embed: embedding matrix of all entities
        :param qualifier_ent:
        :param qualifier_rel:
        :param qual_index:
        :param source_index:
        :return:
        """
        weight = getattr(self, 'w_{}'.format(mode))

        if self.p['STATEMENT_LEN'] != 3:
            # add code here
            if mode != 'loop':
                rel_emb = self.update_rel_emb_with_qualifier(ent_embed, rel_embed, qualifier_ent,
                                                                 qualifier_rel, edge_type, qual_index) #(190348,200)
            else:
                rel_emb = torch.index_select(rel_embed, 0, edge_type)
        else:
            rel_emb = torch.index_select(rel_embed, 0, edge_type)

        xj_rel = self.rel_transform(x_j, rel_emb) #(190348,200)
        out = torch.einsum('ij,jk->ik', xj_rel, weight)#矩阵乘法 (190348,200) * (200,200) => (190348,200)

        if self.p['STAREARGS']['ATTENTION'] and mode != 'loop':# False
            out = out.view(-1, self.heads, self.attn_dim)
            x_i = x_i.view(-1, self.heads, self.attn_dim)

            alpha = torch.einsum('bij,kij -> bi', [torch.cat([x_i, out], dim=-1), self.att])
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, source_index, ent_embed.size(0))
            alpha = F.dropout(alpha, p=self.attn_drop)
            return out * alpha.view(-1, self.heads, 1)
        else:
            return out if edge_norm is None else out * edge_norm.view(-1, 1) #edge_norm:(190348,):[0.0000, 0.0000, 0.0000,  ..., 0.0471, 0.0000, 0.0234] => (190348,1) ; (190348,200) * (190348,1) => (190348,200)

    def update(self, aggr_out, mode):
        if self.p['STAREARGS']['ATTENTION'] and mode != 'loop':# mode:'in' ; aggr_out:(47156,200)
            aggr_out = aggr_out.view(-1, self.heads * self.attn_dim)# aggr_out:(47156,200)

        return aggr_out

    @staticmethod
    def compute_norm(edge_index, num_ent): # num_ent:47156 ; edge_index:tensor(2,190348)
        """
        Re-normalization trick used by GCN-based architectures without attention.

        Yet another torch scatter functionality. See coalesce_quals for a rough idea.

        row         :      [1,1,2,3,3,4,4,4,4, .....]        (about 61k for Jf17k)
        edge_weight :      [1,1,1,1,1,1,1,1,1,  ....] (same as row. So about 61k for Jf17k)
        deg         :      [2,1,2,4,.....]            (same as num_ent about 28k in case of Jf17k)

        :param edge_index:
        :param num_ent:
        :return:
        """
        row, col = edge_index
        edge_weight = torch.ones_like( # (190348,):tensor([1., 1., 1.,  ..., 1., 1., 1.])
            row).float()  # Identity matrix where we know all entities are there
        deg = scatter_add(edge_weight, row, dim=0, #(47156,):tensor([ 0.,  0., 31.,  ...,  8.,  0.,  0.])
                          dim_size=num_ent)  # Summing number of weights of
        # the edges, D = A + I
        deg_inv = deg.pow(-0.5)  # D^{-0.5} # (47156,):tensor([   inf,    inf, 0.1796,  ..., 0.3536,    inf,    inf])
        deg_inv[deg_inv == float('inf')] = 0  # for numerical stability #(47156,):tensor([0.0000, 0.0000, 0.1796,  ..., 0.3536, 0.0000, 0.0000])
        norm = deg_inv[row] * edge_weight * deg_inv[ # (190348,):tensor([0.0000, 0.0000, 0.0000,  ..., 0.0471, 0.0000, 0.0234])
            col]  # Norm parameter D^{-0.5} *

        return norm

    def coalesce_quals(self, qual_embeddings, qual_index, num_edges, fill=0):
        """
         qual_embeddings:(37433,200), qual_index:(37433,):tensor([     3,      3,      6,  ..., 190310, 190324, 190342]) ; num_edges:190348, fill=0
        before:
            qualifier_emb      :   [a,b,c,d,e,f,g,......]               (here a,b,c ... are of 200 dim)
            qual_index         :   [1,1,2,1,2,3,2,......]               (here 1,2,3 .. are edge index of Main COO)
            edge_type  190348  :   [0,0,0,0,0,0,0, .....]               (empty array of size num_edges)

        After:
            edge_type          :   [a+b+d,c+e+g,f ......]        (here each element in the list is of 200 dim)

        :param qual_embeddings: shape of [1, N_QUALS]
        :param qual_index: shape of [1, N_QUALS] which states which quals belong to which main relation from the index,
            that is, all qual_embeddings that have the same index have to be summed up
        :param num_edges: num_edges to return the appropriate tensor
        :param fill: fill value for the output matrix - should be 0 for sum/concat and 1 for mul qual aggregation strat
        :return: [1, N_EDGES]
        """

        if self.p['STAREARGS']['QUAL_N'] == 'sum':
            output = scatter_add(qual_embeddings, qual_index, dim=0, dim_size=num_edges) # (190348,200)
        elif self.p['STAREARGS']['QUAL_N'] == 'mean':
            output = scatter_mean(qual_embeddings, qual_index, dim=0, dim_size=num_edges)

        if fill != 0:
            # by default scatter_ functions assign zeros to the output, so we assign them 1's for correct mult
            mask = output.sum(dim=-1) == 0
            output[mask] = fill

        return output

    def __repr__(self):
        return '{}({}, {}, num_rels={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_rels)
