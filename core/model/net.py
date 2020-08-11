from core.model.net_utils import FC, MLP, LayerNorm
from math import sqrt
from core.model.mca import MCA_ED, MHAtt, FFN, AGAttention  #, BiAttention

import torch.nn as nn
import torch.nn.functional as F
import torch

import warnings
warnings.filterwarnings("ignore")


class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted, att.squeeze()



class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()

        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)

        self.bi_attention = BiAttention(
            __C,
            __C.HIDDEN_SIZE,
            __C.MAX_TOKEN,
            __C.IMG_FEAT_PAD_SIZE
        )
        self.h_fh = nn.Linear(__C.HIDDEN_SIZE, __C.FLAT_OUT_SIZE)

        self.ag_attention = AGAttention(
            __C
        )

        self.fh_h = nn.Linear(__C.FLAT_OUT_SIZE, __C.HIDDEN_SIZE)

    def forward(self, img_feat, ques_ix):

        # Make mask
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)

        # Pre-process Language Feature
        lang_feat = self.embedding(ques_ix)
        lang_feat_ori, _ = self.lstm(lang_feat)

        # Pre-process Image Feature
        img_feat_ori = self.img_feat_linear(img_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat_ori,
            img_feat_ori,
            lang_feat_mask,
            img_feat_mask
        )

        proj_feat, attn_weight = self.bi_attention(img_feat, lang_feat, img_feat_mask, lang_feat_mask)

        lang_feat, lang_weight = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )

        recon_weight = self.ag_attention(img_feat_ori, proj_feat+self.fh_h(lang_feat), img_feat_mask)

        if self.__C.model_type == 'recon_e':
            entropy_rate = self.com_recon_ent_rate(img_feat_mask, attn_weight, self.__C.entropy_tho)
            recon_loss = self.recon_loss_enhance(attn_weight=attn_weight, recon_weight=recon_weight, entropy_rate=entropy_rate)
        else:
            recon_loss = self.recon_loss(attn_weight=attn_weight, recon_weight=recon_weight)

        proj_feat = self.h_fh(proj_feat)
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))

        return proj_feat, recon_loss

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def com_recon_ent_rate(self, img_mask, learned_weight, recon_thod):
        
        # Make mask
        img_mask_tmp = img_mask.squeeze(1).squeeze(1)
        # print(img_mask_tmp[0])
        object_num = 100 - torch.sum(img_mask_tmp, dim=-1)
        # print(object_num[0])
        avg_weight = torch.div(torch.ones_like(learned_weight).float().cuda(), object_num.unsqueeze(1).float())
        # print(avg_weight[0])
        avg_weight = avg_weight.masked_fill(img_mask_tmp, 1e-9)
        # print(avg_weight[0])
        entropy_avg = self.get_entropy(avg_weight)
        # print(entropy_avg, entropy_avg.size())

        entropy_attn = self.get_entropy(torch.where(learned_weight==0.0, torch.zeros_like(learned_weight).float().cuda()+1e-9, learned_weight))
        # print(entropy_attn, entropy_attn.size())

        entropy_rate = torch.where(
            torch.div(entropy_avg-entropy_attn, entropy_avg)>recon_thod, 
            torch.ones_like(entropy_avg).float().cuda(), 
            torch.zeros_like(entropy_avg).float().cuda()
            )
        # print(entropy_rate)

        return entropy_rate


    def get_entropy(self, data_df, columns=None):
        return torch.sum(
            (-data_df)*torch.log(data_df), 
            dim=-1
            )

    def recon_loss_enhance(self, attn_weight, recon_weight, entropy_rate):
        error = (attn_weight - recon_weight).view(attn_weight.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1)  # * 0.0005

        error = torch.sum(torch.mul(error, entropy_rate), dim=-1)

        return error

    def recon_loss(self, attn_weight, recon_weight):
        error = (attn_weight - recon_weight).view(attn_weight.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1)  # * 0.0005

        # Average over batch
        error = error.mean()# * self.__C.recon_rate
        return error


class BiAttention(nn.Module):
    def __init__(self, __C, common_hidden, q_len, i_len):
        super(BiAttention, self).__init__()

        self.linear_q_self_a = nn.Linear(common_hidden, 1)
        self.cos = torch.nn.CosineSimilarity(dim=3)

        self.__C = __C

        self.common_hidden = common_hidden

        self.l_flatten = AttFlat(__C)
        self.i_flatten = AttFlat(__C)

        self.fh_h = nn.Linear(1024, 512)
    def forward(self, i_batch, q_batch, i_mask, q_mask):

        c_len = i_batch.size(1)

        batch_size = q_batch.size(0)

        i_feat, _ = self.qkv_attention(i_batch, q_batch, q_batch, mask=q_mask)
        i_feat, i_weight = self.l_flatten(i_feat, i_mask)

        l_feat, _ = self.qkv_attention(q_batch, i_batch, i_batch, mask=i_mask)
        l_feat, _ = self.i_flatten(l_feat, q_mask)

        return self.fh_h(l_feat + i_feat), i_weight

    def qkv_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2,-1)) / sqrt(d_k)
        if mask is not None:
            scores.data.masked_fill_(mask.squeeze(1), -65504.0)
        
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

