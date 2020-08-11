
from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math

from math import sqrt

import warnings
warnings.filterwarnings("ignore")


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask, get_weight=False):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        if self.__C.model_type == 'base+recon' and get_weight:
            weight, atted = self.att(v, k, q, mask, get_weight)
            atted = atted.transpose(1, 2).contiguous().view(
                n_batches,
                -1,
                self.__C.HIDDEN_SIZE
            )

            atted = self.linear_merge(atted)
            return weight, atted

        atted = self.att(v, k, q, mask, get_weight)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask, get_weight=False):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        if self.__C.model_type == 'base+recon' and get_weight:
            return att_map, torch.matmul(att_map, value)
        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

        self.__C = __C

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        if self.__C.model_type == 'base+recon':
            weight, ret = self.mhatt2(y, y, x, y_mask, get_weight=True)
            x = self.norm2(x + self.dropout2(
                ret
            ))

            x = self.norm3(x + self.dropout3(
                self.ffn(x)
            ))

            return x, weight

        else:
            x = self.norm2(x + self.dropout2(
                self.mhatt2(y, y, x, y_mask)
            ))

            x = self.norm3(x + self.dropout3(
                self.ffn(x)
            ))

            return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

        self.__C = __C

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            x = enc(x, x_mask)

        for dec in self.dec_list:

            if self.__C.model_type == 'base+recon':
                y, weight = dec(y, x, y_mask, x_mask)
            else:
                y = dec(y, x, y_mask, x_mask)

        if self.__C.model_type == 'base+recon':
            return x, y, weight
        else:
            return x, y


class BiAttention(nn.Module):
    def __init__(self, __C, common_hidden, q_len, i_len):
        super(BiAttention, self).__init__()

        self.linear_q_self_a = nn.Linear(common_hidden, 1)
        self.cos = torch.nn.CosineSimilarity(dim=3)

        self.__C = __C

        self.common_hidden = common_hidden

        self.convert_weight_c = nn.Sequential(
            nn.Linear(q_len * i_len, i_len),
            nn.BatchNorm1d(i_len),
            nn.LeakyReLU(),
            nn.Linear(i_len, i_len)
        )
        self.convert_weight_q = nn.Sequential(
            nn.Linear(q_len*i_len, q_len),
            nn.BatchNorm1d(q_len),
            nn.LeakyReLU(),
            nn.Linear(q_len, q_len)
        )

        self.que_list = nn.ModuleList([SA(__C) for _ in range(self.__C.GA_layer)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(self.__C.GA_layer)])

        self.c_len = i_len
        self.q_len = q_len
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, i_batch, q_batch, i_mask, q_mask):

        c_len = i_batch.size(1)

        batch_size = q_batch.size(0)

        # i_batch_ = i_batch.unsqueeze(2).expand(batch_size, c_len, self.q_len, self.common_hidden)
        # q_batch_ = q_batch.unsqueeze(1).expand(batch_size, c_len, self.q_len, self.common_hidden)

        # cq = self.cos(i_batch_, q_batch_)


        cq = torch.matmul(i_batch, q_batch.transpose(-2, -1)) / sqrt(self.common_hidden)

        # torch.matmul(i_batch, q_batch.transpose(-2, -1)) / sqrt(common_hidden)

        # add_mask
        i_mask_ = i_mask.squeeze().unsqueeze(2).float()
        q_mask_ = q_mask.squeeze().unsqueeze(1).float()
        # print("i_mask_", i_mask_.size())
        # print("q_mask_", q_mask_.size())
        cq_mask = torch.matmul(i_mask_, q_mask_)
        # print("cq_mask_", cq_mask)
        cq = cq.masked_fill(cq_mask.byte(), 0)

        e_dis_i = self.convert_weight_c(cq.view(batch_size, -1))
        e_dis_q = self.convert_weight_q(cq.view(batch_size, -1))

        # add mask
        e_dis_i = e_dis_i.masked_fill(i_mask.squeeze(), 0)
        e_dis_q = e_dis_q.masked_fill(q_mask.squeeze(), 0)

        a_dis_i = F.softmax(e_dis_i)
        a_dis_q = F.softmax(e_dis_q)

        # apply attention features
        attented_v = torch.sum(
            torch.mul(i_batch, a_dis_i.unsqueeze(2). expand(batch_size, c_len, self.common_hidden))
            , dim=1)
        attented_q = torch.sum(
            torch.mul(
                q_batch, a_dis_q.unsqueeze(2). expand(batch_size, self.q_len, self.common_hidden)
            )
            , dim=1)

        return attented_v + attented_q, a_dis_i


class AGAttention(nn.Module):
    def __init__(self, __C):
        super(AGAttention, self).__init__()
        self.lin_v = FFN(__C)  # let self.lin take care of bias
        self.lin_q = FFN(__C)
        self.lin = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=1,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, v, q, v_mask):
        """
        v = batch, num_obj, dim
        q = batch, dim
        """
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)

        x = v * q
        x = self.lin(x)  # batch, num_obj, glimps

        x = x.squeeze(-1).masked_fill(v_mask.squeeze(2).squeeze(1), -1e9)

        x = F.softmax(x, dim=1)

        # x = self.drop(x)
        return x
