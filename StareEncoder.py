import torch
import numpy as np

from typing import Dict
from torch import nn
from torch.nn import Parameter
from utils.utils_gcn import get_param
from models.gnn_layer import StarEConvLayer
from transformers import AutoConfig
from models.bert_for_layerwise import BertModelForLayerwise
from models.prompter import Prompter
from helper import get_performance, get_loss_fn, GRAPH_MODEL_CLASS


class StareEncoder(nn.Module):
    def __init__(self, graph_repr: Dict[str, np.ndarray], config: dict, configs, text_dict, gt):
        super().__init__()

        # CSProm-KG
        self.configs = configs
        self.ent_names = text_dict['ent_names']
        self.rel_names = text_dict['rel_names']
        self.ent_descs = text_dict['ent_descs']
        self.all_tail_gt = gt['all_tail_gt']
        self.all_head_gt = gt['all_head_gt']



        self.graph_model = GRAPH_MODEL_CLASS[self.configs.graph_model](configs)#conve

        self.loss_fn = get_loss_fn(configs)
        self._MASKING_VALUE = -1e4 if self.configs.use_fp16 else -1e9


        # StarE


        self.config = config
        self.act = torch.tanh if 'ACT' not in config['STAREARGS'].keys() \
            else config['STAREARGS']['ACT']
        self.bceloss = torch.nn.BCELoss()
        self.emb_dim = config['EMBEDDING_DIM']
        self.num_rel = config['NUM_RELATIONS']
        self.num_ent = config['NUM_ENTITIES']
        self.n_bases = config['STAREARGS']['N_BASES']
        self.n_layer = config['STAREARGS']['LAYERS']
        self.gcn_dim = config['STAREARGS']['GCN_DIM']
        self.hid_drop = config['STAREARGS']['HID_DROP']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        # self.bias = config['STAREARGS']['BIAS']
        self.model_nm = config['MODEL_NAME'].lower()#stare_transformer
        self.triple_mode = config['STATEMENT_LEN'] == 3#False
        self.qual_mode = config['STAREARGS']['QUAL_REPR']#sparse


        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        self.device = config['DEVICE']

        # Storing the KG
        self.edge_index = torch.tensor(graph_repr['edge_index'], dtype=torch.long, device=self.device)
        self.edge_type = torch.tensor(graph_repr['edge_type'], dtype=torch.long, device=self.device)

        if not self.triple_mode:
            if self.qual_mode == "full":
                self.qual_rel = torch.tensor(graph_repr['qual_rel'], dtype=torch.long, device=self.device)
                self.qual_ent = torch.tensor(graph_repr['qual_ent'], dtype=torch.long, device=self.device)
            elif self.qual_mode == "sparse":
                self.quals = torch.tensor(graph_repr['quals'], dtype=torch.long, device=self.device)

        self.gcn_dim = self.emb_dim if self.n_layer == 1 else self.gcn_dim

        self.init_embed = get_param((self.num_ent, self.emb_dim))
        self.init_embed.data[0] = 0  # padding



        if self.model_nm.endswith('transe'):
            self.init_rel = get_param((self.num_rel, self.emb_dim))
        elif config['STAREARGS']['OPN'] == 'rotate' or config['STAREARGS']['QUAL_OPN'] == 'rotate':#sub
            phases = 2 * np.pi * torch.rand(self.num_rel, self.emb_dim // 2)
            self.init_rel = nn.Parameter(torch.cat([
                torch.cat([torch.cos(phases), torch.sin(phases)], dim=-1),
                torch.cat([torch.cos(phases), -torch.sin(phases)], dim=-1)
            ], dim=0))
        else:
            self.init_rel = get_param((self.num_rel * 2, self.emb_dim))#*

        self.init_rel.data[0] = 0 # padding

        self.conv1 = StarEConvLayer(self.emb_dim, self.gcn_dim, self.num_rel, act=self.act,# 200 200 532
                                       config=config)
        self.conv2 = StarEConvLayer(self.gcn_dim, self.emb_dim, self.num_rel, act=self.act,
                                       config=config) if self.n_layer == 2 else None

        if self.conv1: self.conv1.to(self.device)
        if self.conv2: self.conv2.to(self.device)

        self.register_parameter('bias', Parameter(torch.zeros(self.num_ent)))

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    #sub:Tensor(8,):tensor([ 1622,  3095, 28392, 13803, 21893, 46724, 29324, 29136],device='cuda:0')
    #rel:Tensor(8,):tensor([  63,  213,   14, 1006,  116,  552,  763,  127], device='cuda:0')
    def get_s_r_x_emb(self, sub, rel, quals=None, embed_qualifiers: bool = False, return_mask: bool = False):
        # sub, rel = sub_rel
        # r = self.init_rel if not self.model_nm.endswith('transe') else torch.cat([self.init_rel, -self.init_rel], dim=0)

        r = self.init_rel

        # sparse

        # x=init_embed:tensor(47156,200), edge_index:(2,380696), edge_type:(380696,), rel_embed:(1064,200), qual_ent:, qual_rel,quals:(3,74866)
        x, r = self.conv1(x=self.init_embed, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=None,
                                  qualifier_rel=None,
                                  quals=self.quals)

        x = self.hidden_drop(x)
        x, r = self.conv2(x=x, edge_index=self.edge_index,
                                  edge_type=self.edge_type, rel_embed=r,
                                  qualifier_ent=None,
                                  qualifier_rel=None,
                                  quals=self.quals) if self.n_layer == 2 else (x, r)

        x = self.feature_drop(x) if self.n_layer == 2 else x
        # 得到聚合后的头实体嵌入和关系嵌入

        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb,rel_emb, x

    def forward(self, ent, rel, quals, src_ids, src_mask):
        bs = ent.size(0)#128
        ent_embed, rel_embed, all_ent_embed = self.get_s_r_x_emb(ent, rel, quals, True,True)
        # ent : tensor(128,)
        # rel : tensor(128,)
        # quals : tensor(128,12)
        # src_ids : tensor(128,42)
        # src_mask : tensor(128,42)
        # ent_embed : tensor(128,156)
        # rel_embed : tensor(128,156)
        # all_ent_embed : tensor(47156,156)

        #
        #

        # pred -- .shape: (batch_size, embed_dim)
        pred = self.graph_model(ent_embed, rel_embed)#(128,156)
        # logits -- .shape: (batch_size, n_ent)
        logits = self.graph_model.get_logits(pred, all_ent_embed)#(128,47156)
        return logits, pred