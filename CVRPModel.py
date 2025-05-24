import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import revtorch as rv
import numpy as np
import util

import logging
logger = logging.getLogger(__name__)
import math
from einops import rearrange

from timm.models.layers import DropPath, trunc_normal_

class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        # self.encoder = CVRP_Encoder(**model_params)
        self.encoder = Rev_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        #node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_colors = reset_state.node_colors
        #node_xy_demand = torch.cat((node_xy, node_colors), dim=2)
        # shape: (batch, problem, 3)

        self.encoded_nodes = self.encoder(depot_xy, node_xy, node_colors)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes)

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)


        if state.selected_count == 0:  # First Move, depot
            selected = torch.zeros(size=(batch_size, pomo_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, pomo_size))

            # # Use Averaged encoded nodes for decoder input_1
            # encoded_nodes_mean = self.encoded_nodes.mean(dim=1, keepdim=True)
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q1(encoded_nodes_mean)

            # # Use encoded_depot for decoder input_2
            # encoded_first_node = self.encoded_nodes[:, [0], :]
            # # shape: (batch, 1, embedding)
            # self.decoder.set_q2(encoded_first_node)

        elif state.selected_count == 1:  # Second Move, POMO
            selected = torch.arange(start=1, end=pomo_size+1)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs = self.decoder(encoded_last_node, state.pre_step_color, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem+1)

            if self.training or self.model_params['eval_type'] == 'softmax':
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    if (prob != 0).all():
                        break

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None  # value not needed. Can be anything.

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes

########################################
# Rev_MHA_FFN_ENCODER
########################################

class Rev_Encoder(nn.Module):
    # def __init__(
    #     self,
    #     n_layers: int,  # encoder_layer_num
    #     n_heads: int,   # head_num
    #     embedding_dim: int,   # embedding_dim
    #     input_dim: int,       # /
    #     intermediate_dim: int,    # ff_hidden_dim
    #     add_init_projection=True, # /
    # ):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        head_num = self.model_params['head_num']
        intermediate_dim = self.model_params['ff_hidden_dim']
        qkv_dim = self.model_params['qkv_dim']
        # if add_init_projection or input_dim != embedding_dim:
        #     self.init_projection_layer = torch.nn.Linear(input_dim, embedding_dim)
        # 初始映射 inital projection: depot + node + color
        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(2, embedding_dim)
        self.embedding_colors = nn.Linear(4, embedding_dim)
        self.embedding_depot_colors = nn.Linear(4, embedding_dim)

        self.num_hidden_layers = encoder_layer_num
        blocks = []
        for _ in range(encoder_layer_num):
            # f_func = MHABlock(embedding_dim, head_num)
            # f_func = EncoderLayer(embedding_dim, head_num, qkv_dim)
            f_func = MLLABlock(embedding_dim, head_num, qkv_bias=True, drop_path=0.)
            # f_func = NormLinearAttention(embedding_dim, intermediate_dim, head_num)
            # g_func = FFBlock(embedding_dim, intermediate_dim)
            # g_func_mid = MSCFFN_step1(embedding_dim, intermediate_dim)
            # g_func = MSCFFN_step2(g_func_mid, embedding_dim, intermediate_dim)
            # g_func = MSCFFN(embedding_dim, intermediate_dim)
            g_func = GLU(embedding_dim, intermediate_dim, act_fun=F.sigmoid)
            # we construct a reversible block with our F and G functions
            blocks.append(rv.ReversibleBlock(f_func, g_func, split_along_dim=-1))

        self.sequence = rv.ReversibleSequence(nn.ModuleList(blocks))

    # def forward(self, x, mask=None):
    def forward(self, depot_xy, node_xy, node_colors):
        # if hasattr(self, "init_projection_layer"):
        #     x = self.init_projection_layer(x)

        batch_size = depot_xy.shape[0]
        depot_color=torch.ones((batch_size, 1, 4),device=depot_xy.device)
        embedded_depot_color = self.embedding_depot_colors(depot_color.float())
        embedded_depot = self.embedding_depot(depot_xy.float())
        embedded_depot = embedded_depot_color+embedded_depot
        # shape: (batch, 1, embedding)
        #颜色和坐标编码
        embedded_node_xy = self.embedding_node(node_xy.float())
        embedded_node_colors = self.embedding_colors(node_colors.float())
        embedded_node = embedded_node_xy+embedded_node_colors
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)
    
        out = torch.cat([out, out], dim=-1)
        out = self.sequence(out)
        return torch.stack(out.chunk(2, dim=-1))[-1]





########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(2, embedding_dim)
        self.embedding_colors = nn.Linear(4, embedding_dim)
        self.embedding_depot_colors = nn.Linear(4, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy, node_colors):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)
        batch_size = depot_xy.shape[0]  # 或者通过其他方式获取 batch_size
        # pomo_size = node_colors.shape[1]
        depot_color=torch.ones((batch_size, 1, 4),device=depot_xy.device)
        embedded_depot_color = self.embedding_depot_colors(depot_color.float())
        embedded_depot = self.embedding_depot(depot_xy.float())
        embedded_depot = embedded_depot_color+embedded_depot
        # shape: (batch, 1, embedding)
        #颜色和坐标编码
        embedded_node_xy = self.embedding_node(node_xy.float())
        embedded_node_colors = self.embedding_colors(node_colors.float())
        embedded_node = embedded_node_xy+embedded_node_colors
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        for layer in self.layers:
            out = layer(out)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim: int, head_num: int, qkv_dim: 16):
        super().__init__()
        
        # self.model_params = model_params
        # embedding_dim = self.model_params['embedding_dim']
        # head_num = self.model_params['head_num']
        # qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        # self.feed_forward = FeedForward(**model_params)
        # self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        out_concat = multi_head_attention(q, k, v)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        # out1 = self.add_n_normalization_1(input1, multi_head_out)
        # out2 = self.feed_forward(out1)
        # out3 = self.add_n_normalization_2(out1, out2)
        # return out3
    
        return multi_head_out
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        # self.Wq_1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim+4, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        # self.q1 = None  # saved q1, for multi-head attention
        # self.q2 = None  # saved q2, for multi-head attention

    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, pre_step_color, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # load.shape: (batch, pomo)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, pre_step_color), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        # q = self.q1 + self.q2 + q_last
        # # shape: (batch, head_num, pomo, qkv_dim)
        q = q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    return out_concat

class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans

class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
    
class FFBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
    # def __init__(self, **model_params):
        super().__init__()
        # hidden_size = model_params['embedding_dim']
        # intermediate_size = model_params['ff_hidden_dim']

        self.feed_forward = nn.Linear(hidden_size, intermediate_size)
        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_layer_norm = nn.BatchNorm1d(hidden_size)
        self.activation = nn.GELU()
        # self.activation = F.relu()
    # def forward(self, hidden_states: Tensor):
    def forward(self, input1):
        input1 = (
            self.output_layer_norm(input1.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
        )
        intermediate_output = self.feed_forward(input1)
        intermediate_output = self.activation(intermediate_output)
        output = self.output_dense(intermediate_output)

        return output

class MHABlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.mixing_layer_norm = nn.BatchNorm1d(hidden_size)
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, bias=False)

    def forward(self, hidden_states):

        assert hidden_states.dim() == 3
        hidden_states = self.mixing_layer_norm(hidden_states.transpose(1, 2)).transpose(
            1, 2
        )
        hidden_states_t = hidden_states.transpose(0, 1)
        mha_output = self.mha(hidden_states_t, hidden_states_t, hidden_states_t)[
            0
        ].transpose(0, 1)

        return mha_output


########################################
# MSCFFN
########################################

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}

BertLayerNorm = torch.nn.LayerNorm


class MSCFFN_step1(nn.Module):
    def __init__(self, hidden_size, intermediate_dim):
        super(MSCFFN_step1, self).__init__()
        # 指定FFN的子空间个数（类似多头注意力中的多头概念），即把输入特征维度hidden_size按 ffn_head 个子空间进行均分处理
        self.ffn_head = 16
        # 每个子空间的维度大小，即将输入特征维度等分后的单个子空间维度
        self.ffn_head_size = int(hidden_size / self.ffn_head)
        # 首先对输入进行线性映射（从 hidden_size 到 hidden_size）。
        # 这个线性层与普通FFN中的第一层映射相似，但这里是在分子空间处理之前统一做一次变换
        self.dense_ffn_head = nn.Linear(hidden_size, hidden_size)
        # 指定中间扩张维度的大小
        # 普通FFN通常有一个中间层维度（如4 * hidden_size），这里是对每个子空间进行扩张，因此中间维度是 ffn_head_size * m，给定 m=6
        # 对于每个子空间，从 d/n 扩张到 m*(d/n)
        # self.ffn_intermediate_size = self.ffn_head_size * 8
        self.ffn_intermediate_size = intermediate_dim * 2
        # 尺寸为 (ffn_head, ffn_head_size, ffn_intermediate_size) 的参数张量，
        # 对每个子空间从 ffn_head_size 映射到 ffn_intermediate_size
        # 相当于 n 个独立的线性变换权重合在一起（n=ffn_head个子空间，每个子空间一个线性层）。
        self.ffn_intermediate_weights = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(self.ffn_head,
                                                                                          self.ffn_head_size, self.ffn_intermediate_size)))
        # 尺寸为 (ffn_head, ffn_intermediate_size) 的偏置参数，对上述线性变换进行偏置加成
        self.ffn_intermediate_bias = nn.Parameter(nn.init.zeros_(torch.empty(self.ffn_head,
                                                                             self.ffn_intermediate_size)))
        # self.dropout = nn.Dropout(dropout_rate)
        # 定义激活函数，这里使用ReLU激活（ACT2FN是一个映射函数名到具体激活函数的字典）
        self.relu_act = ACT2FN["relu"]

    def transpose_for_scores_original(self, x):
        x_size = x.size()
        new_x_shape = [x_size[0]*x_size[1], self.ffn_head, self.ffn_head_size]
        x = x.view(*new_x_shape)
        return x.permute(1, 0, 2)
    def transpose_for_scores(self, x):
        B, L, D = x.size()
        assert D % self.ffn_head == 0, f"Hidden size {D} must be divisible by number of heads {self.ffn_head}"
        self.ffn_head_size = D // self.ffn_head  # 确保计算正确
        # shape: (B, L, D) -> (B * L, ffn_head, ffn_head_size) (B*L, n, d/n)  
        new_x_shape = (B * L, self.ffn_head, self.ffn_head_size)
        x = x.view(*new_x_shape)
        # shape: (B * L, ffn_head, ffn_head_size) -> (ffn_head, B * L, ffn_head_size)  (n, B*L, d/n)
        return x.permute(1, 0, 2)

    def forward(self, hidden_states):
        # 对输入先进行一次线性映射，不改变维度，只是进行特征的再组合
        hidden_states = self.dense_ffn_head(hidden_states)
        # 调用前面的函数将张量从 (B, L, D) 转换为 (n, B*L, d/n) 格式，方便对每个子空间进行处理
        hidden_states = self.transpose_for_scores_original(hidden_states)
        # 对每个子空间（维度为 (n, B*L, d/n)) 应用线性变换到中间维度 (m*(d/n))
        # 执行后结果是 (n, B*L, m*(d/n)) 的张量，加上相应的bias
        # self.ffn_intermediate_weights 是 (n, d/n, m*(d/n)), 
        # hidden_states是 (n, B*L, d/n), 
        # 矩阵乘得到 (n, B*L, m*(d/n))
        hidden_states = torch.matmul(hidden_states, self.ffn_intermediate_weights) + self.ffn_intermediate_bias.unsqueeze(1)
        # 将前一半子空间 (即n/2个子空间) 的结果通过ReLU激活
        # hidden_states[0 : n/2, :, :] 取前一半子空间对应的维度
        hidden_relu = self.relu_act(hidden_states[0: int(self.ffn_head/2), :, :])
        # 将后半部分子空间的结果保持线性（不激活）
        hidden_linear = hidden_states[int(self.ffn_head/2):self.ffn_head, :, :]
        # 将前半子空间的ReLU激活结果与后半子空间的线性结果逐元素相乘
        # 这种操作结合了两组子空间的特征，类似于门控机制，增强表示能力
        hidden_states = hidden_relu * hidden_linear
        # hidden_states = self.dropout(hidden_states)
        # 返回 (n/2, B*L, m*(d/n)) 形状的张量，用于下游步骤（如step2）进行维度还原和整合
        return hidden_states


class MSCFFN_step2(nn.Module):
    def __init__(self, g_func_mid: nn.Module, hidden_size: int, intermediate_dim: int):
        super(MSCFFN_step2, self).__init__()
        self.ffn_head = 16
        self.ffn_head_size = int(hidden_size / self.ffn_head)
        # self.ffn_intermediate_size = self.ffn_head_size * 8
        self.ffn_intermediate_size = intermediate_dim * 2
        # shape: (n/2, ffn_intermediate_size, ffn_head_size)
        # 将 step1 输出的子空间 (已经经过非线性组合) 从 m*(d/n) 降回 d/n 的线性变换的权重参数
        # 这里每个子空间对 C_i 都有一组降维权重，共有 n/2 个 C_i
        self.ffn_weights = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(int(self.ffn_head/2),
                                                                             self.ffn_intermediate_size, self.ffn_head_size)))
        # shape: (n/2, ffn_head_size)
        # 对上述线性变换加上偏置项
        self.ffn_bias = nn.Parameter(nn.init.zeros_(torch.empty(int(self.ffn_head/2),
                                                                self.ffn_head_size)))
        # 最终的线性映射，将所有 n/2 个子空间处理后的结果拼接，形成 (B, L, d/2)，再映射回 (B, L, d)
        # 最终该线性层的权重维度为 (d, d/2)，作用是 d/2 -> d
        self.dense2 = nn.Linear(int(hidden_size/2), hidden_size)
        # 对最终输出进行层归一化
        self.LayerNorm = BertLayerNorm(hidden_size, eps=1e-05)
        # self.LayerNorm = nn.BatchNorm1d(hidden_size)
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 对最终输出进行随机失活，以防止过拟合
        self.dropout = nn.Dropout(0.1)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加 input_tensor 的占位变量
    #     self.input_tensor = None
    #
    # def set_input_tensor(self, input_tensor: torch.Tensor):
    #     """设置输入张量"""
    #     self.input_tensor = input_tensor

    def forward(self, hidden_states, input_tensor):
        """完成 step2 的前向传播"""
        # if self.input_tensor is None:
        #     raise ValueError("******MSCFFN2_input_tensor_Input tensor not set. Call `set_input_tensor` before forward.******")

        input_tensor = input_tensor
        input_tensor_size = input_tensor.size()
        # 在step1的输出中，hidden_states是 (n/2, B*L, m*(d/n))
        # hidden_states: (n/2, B*L, m*(d/n))
        # self.ffn_weights: (n/2, ffn_intermediate_size, ffn_head_size)
        # 矩阵相乘后，结果 (n/2, B*L, d/n), 加上 bias 后仍是(n/2, B*L, d/n)
        hidden_states = torch.matmul(hidden_states, self.ffn_weights) + self.ffn_bias.unsqueeze(1)
        hidden_states = torch.reshape(hidden_states.permute(1, 0, 2), [input_tensor_size[0], input_tensor_size[1], self.ffn_head_size*int(self.ffn_head/2)])
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MSCFFN(nn.Module):
    def __init__(self, embedding_dim, intermediate_dim):
        super(MSCFFN, self).__init__()
        self.step1 = MSCFFN_step1(embedding_dim, intermediate_dim)
        self.step2 = MSCFFN_step2(self.step1, embedding_dim, intermediate_dim)

    def forward(self, input_tensor):
        # input_tensor 是同时传递给 step1 和 step2 的输入
        step1_output = self.step1(input_tensor)
        # step2 接受 step1 的输出和原始 input_tensor
        output = self.step2(step1_output, input_tensor)
        return output
    
########################################
# MLLA
########################################

class LinearAttentionforCTSP(nn.Module):
    """ Simplified Linear Attention.

    Args:
        dim (int): Number of input channels (embedding dimension).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 缩放因子，用于稳定训练
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 生成 Q, K, V
        self.out_proj = nn.Linear(dim, dim)  # 输出投影

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, sequence_length, embedding_dim)
        Returns:
            Tensor with shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, sequence_length, embedding_dim = x.shape
        assert embedding_dim == self.dim, "Input embedding_dim must match initialized dim"

        # Step 1: Compute Q, K, V
        qkv = self.qkv(x)  # Shape: (batch_size, sequence_length, 3 * embedding_dim)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # Shapes: (batch_size, num_heads, sequence_length, head_dim)

        # Step 2: Linear Attention
        # Apply softmax along sequence_length for K
        k = k.softmax(dim=-2)  # Normalize along sequence length
        q = q * self.scale  # Scale Q

        # Compute attention scores: (batch_size, num_heads, sequence_length, sequence_length)
        context = torch.einsum("bhlk,bhlv->bhlv", k, v)  # Weighted sum of V
        x = torch.einsum("bhlv,bhlk->bhlv", q, context)  # Q * Context

        # Step 3: Reshape and project output
        x = x.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, embedding_dim)  # Combine heads
        x = self.out_proj(x)  # Final linear projection

        return x

class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    # def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
    #              act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
    def __init__(self, dim, num_heads, qkv_bias=True, drop_path=0., **kwargs):
        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.num_heads = num_heads
        # self.mlp_ratio = mlp_ratio

        # self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        # self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.act = nn.SiLU()
        # self.act = nn.ReLU()
        self.act = get_activation_fn("leaky_relu")
        # self.attn = LinearAttentionforCTSP(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        # self.attn = AgentAttentionforCTSP(dim=dim, num_heads=num_heads)
        self.attn = NormLinearAttention(embed_dim=dim, hidden_dim=dim*4, num_heads=num_heads)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        # self.norm2 = norm_layer(dim)
        # self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        # H, W = self.input_resolution
        # B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        # x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x

        # x = self.norm1(x)
        # step1 - right_side: Linear + ReLU
        act_res = self.act(self.act_proj(x))
        # x = self.in_proj(x).view(B, H, W, C)
        # step1 - left_side: Linear + (Conv) + ReLU
        x = self.in_proj(x)
        # x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)
        x = self.act(x)

        # step2 - Linear Attention
        # 生成 Q, K, V：通过线性层将输入特征映射为查询、键和值, 生成 Q, K, V：通过线性层将输入特征映射为查询、键和值
        x = self.attn(x)

        # step3 - step2 - Linear Attention * step1 - right_side
        x = self.out_proj(x * act_res)
        # 随机深度模块 Stochastic Depth, 在训练期间以一定概率丢弃输入数据，从而增强模型的正则化效果
        x = self.drop_path(x)
        x = shortcut + self.drop_path(x)
        # x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"mlp_ratio={self.mlp_ratio}"



########################################
# Norm Linear Attention (for CTSP)
# Code: https://github.com/Doraemonzzz/transnormer-v2-pytorch/blob/main/transnormer_v2/norm_linear_attention.py
# Paper: https://www.semanticscholar.org/reader/e3fc46d5f4aae2c7a8a86b6bd21ca8db5d40fcbd
########################################

def get_activation_fn(activation):
    logger.info(f"activation: {activation}")
    if activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "elu":
        return F.elu
    elif activation == "sigmoid":
        return F.sigmoid
    elif activation == "exp":
        return torch.exp
    elif activation == "leak":
        return F.leaky_relu
    elif activation == "1+elu":
        def f(x):
            # 调用自定义函数F.elu对x进行处理
            # F.elu是一个自定义函数，可能用于计算指数线性单元（ELU）激活函数
            return 1 + F.elu(x)
        return f
    elif activation == "2+elu":
            def f(x):
                return 2 + F.elu(x)
            return f
    elif activation == "silu":
        return F.silu
    elif activation == "Leaky ReLU":
        return lambda x: F.leaky_relu(x, negative_slope=0.2)
    else:
        return lambda x: x
    
def get_norm_fn(norm_type):
    if norm_type == "layernorm":
        return nn.LayerNorm
    elif norm_type == "batchnorm":
        return nn.BatchNorm1d
    elif norm_type == "instancenorm":
        return nn.InstanceNorm1d
    else:
        # 默认返回 layernorm 或者不做处理
        return nn.BatchNorm1d


class NormLinearAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        act_fun="leaky_relu",
        uv_act_fun="silu",
        # act_fun="elu",
        # uv_act_fun="swish",
        norm_type="instancenorm", # optional: layernorm / batchnorm / instancenorm
        causal=False,
    ):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        self.u_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = get_activation_fn(act_fun)
        self.uv_act = get_activation_fn(uv_act_fun)
        self.num_heads = num_heads
        # self.norm = get_norm_fn(norm_type)(hidden_dim)
        self.norm_type = norm_type
        NormLayer = get_norm_fn(norm_type)
        if norm_type == "layernorm":
            # LayerNorm 通常直接传维度 hidden_dim 即可
            self.norm = NormLayer(hidden_dim)
        else:
            # BatchNorm1d / InstanceNorm1d 对“通道数”做归一化，这里相当于 hidden_dim 为 channel
            self.norm = NormLayer(hidden_dim, affine=True)
        self.causal = causal
        
    def forward(
        self,
        x,
        y=None,
        attn_mask=None,
    ):
        # x: b n d
        if y == None:
            y = x
        n = x.shape[-2]
        # linear map
        q = self.q_proj(x)
        # u = self.u_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        # uv act
        u = self.uv_act(u)
        v = self.uv_act(v)
        # reshape
        q, k, v = map(lambda x: rearrange(x, '... n (h d) -> ... h n d', h=self.num_heads), [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)
        
        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.tril(torch.ones(n, n))).to(q)
            l1 = len(q.shape)
            l2 = len(attn_mask.shape)
            for _ in range(l1 - l2):
                attn_mask = attn_mask.unsqueeze(0)
            energy = torch.einsum('... n d, ... m d -> ... n m', q, k)
            energy = energy * attn_mask
            output = torch.einsum('... n m, ... m d -> ... n d', energy, v)
        else:
            kv = torch.einsum('... n d, ... n e -> ... d e', k, v)
            output = torch.einsum('... n d, ... d e -> ... n e', q, kv)
        # reshape
        output = rearrange(output, '... h n d -> ... n (h d)')
        # --- norm ---
        # If layernorm, we can do self.norm(output) directly:
        if self.norm_type == "layernorm":
            output = self.norm(output)
        else:
            # If batchnorm / instancenorm, need (B, C, L) format
            # 需要先把 (B, N, hidden_dim) -> (B, hidden_dim, N) 才能直接用 *Norm1d
            output = output.transpose(1, 2)  # -> (B, hidden_dim, N)
            output = self.norm(output)
            output = output.transpose(1, 2)  # -> (B, N, hidden_dim)
        # # normalize
        # output = self.norm(output)
        # gate
        # output = u * output
        # outproj
        output = self.out_proj(output)

        return output

########################################
# GLU -> FFN    
########################################

class GLU(nn.Module):
    def __init__(self, d1, d2, act_fun, fina_act="None", dropout=0.0, bias=True):
        super().__init__()
        self.l1 = nn.Linear(d1, d2, bias=bias)
        self.l2 = nn.Linear(d1, d2, bias=bias)
        self.l3 = nn.Linear(d2, d1, bias=bias)
        self.act_fun = get_activation_fn(act_fun)
        self.p = dropout
        if self.p > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        self.fina_act = get_activation_fn(fina_act)

    def forward(self, x):
        o1 = self.l1(x)
        weight = self.act_fun(o1)
        if self.p > 0.0:
            weight = self.dropout(weight)
        o2 = self.l2(x)
        output = weight * o2
        output = self.l3(output)
        output = self.fina_act(output)

        return output




######################################################
# Agent Attention MaxPooling doesn't work in Encoder
######################################################

class AgentAttentionforCTSP(nn.Module):
    r"""
    Simplified Agent Attention for input shape (batch_size, sequence_length, embedding_dim).
    Args:
        dim (int): Number of input channels (embedding dimension).
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weights. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output projection. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # Combine Q, K, V projections
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # Output projection
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            Output tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        batch_size, sequence_length, embedding_dim = x.shape
        assert embedding_dim == self.dim, "Input embedding_dim must match initialized dim"

        # Step 1: Compute Q, K, V
        qkv = self.qkv(x)  # Shape: (batch_size, sequence_length, 3 * embedding_dim)
        qkv = qkv.view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # Shapes: (3, batch_size, num_heads, sequence_length, head_dim)

        # Step 2: Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, sequence_length, sequence_length)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Step 3: Apply attention to values
        x = (attn @ v)  # (batch_size, num_heads, sequence_length, head_dim)
        x = x.transpose(1, 2).reshape(batch_size, sequence_length, embedding_dim)  # Combine heads

        # Step 4: Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
########################################
# class AgentAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  shift_size=0, agent_num=49, **kwargs):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)
#         self.shift_size = shift_size

#         self.agent_num = agent_num
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0], 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1]))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         b, n, c = x.shape
#         h = int(n ** 0.5)
#         w = int(n ** 0.5)
#         num_heads = self.num_heads
#         head_dim = c // num_heads
#         qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#         # q, k, v: b, n, c

#         agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
#         q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

#         position_bias1 = nn.functional.interpolate(self.an_bias, size=self.window_size, mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v

#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v

#         x = x.transpose(1, 2).reshape(b, n, c)
#         v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops