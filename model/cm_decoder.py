import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from model.transformer_decoder import Decoder
from model.transformer_encoder import Encoder
from model.layers import PositionwiseFeedForward

class T2Vdecoder(nn.Module):
    def __init__(self, configs):
        super(T2Vdecoder, self).__init__()
        self.wq = nn.Linear(configs.d_model, configs.d_model)
        self.wk = nn.Linear(configs.d_model, configs.d_model)
        self.wv = nn.Linear(configs.d_model, configs.d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.d_model = configs.d_model
        self.n_head = configs.num_heads
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.ffn = PositionwiseFeedForward(d_model=configs.d_model, hidden=configs.d_model, drop_prob=configs.drop_rate)

    def forward(self, t_input, v_input): 
        # q,k,v
        q_e, k_w, v_w = self.wq(t_input), self.wk(v_input), self.wv(v_input)
        # to n head
        q_e, k_w, v_w = self.split(q_e), self.split(k_w), self.split(v_w)
        # MatMul
        output = self.cm_attention(q_e, k_w, v_w)
        output = self.concat(output) # (batch, seg_num, d_model)
        output = self.norm1(output + t_input) # add norm
        output_ = self.ffn(output) # ffn
        output = self.norm2(output_ + output)
        return output
    
    def cm_attention(self, q_e, k_w, v_w):
        '''
        q_e evidence as key
        k_w, v_w word as key and value
        '''
        # Q, K ,V
        _, _, _, d_tensor = q_e.size()
        scores = torch.matmul(q_e, k_w.transpose(2, 3)) / math.sqrt(d_tensor)
        p_attn = F.softmax(scores, dim = -1)
        out = torch.matmul(p_attn, v_w)
        # out_dim (batch, seq_len, d_model)
        return out
    
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor
    
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class V2Tdecoder(nn.Module):
    def __init__(self, configs):
        super(V2Tdecoder, self).__init__()
        self.wq = nn.Linear(configs.d_model, configs.d_model)
        self.wk = nn.Linear(configs.d_model, configs.d_model)
        self.wv = nn.Linear(configs.d_model, configs.d_model)
        self.softmax = nn.Softmax(dim=-1)
        self.d_model = configs.d_model
        self.n_head = configs.num_heads
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.ffn = PositionwiseFeedForward(d_model=configs.d_model, hidden=configs.d_model, drop_prob=configs.drop_rate)

    def forward(self, v_input, t_input): 
        # q,k,v
        q_e, k_w, v_w = self.wq(v_input), self.wk(t_input), self.wv(t_input)
        # to n head
        q_e, k_w, v_w = self.split(q_e), self.split(k_w), self.split(v_w)
        # MatMul
        output = self.cm_attention(q_e, k_w, v_w)
        output = self.concat(output) # (batch, seg_num, d_model)
        output = self.norm1(output + v_input) # add norm
        output_ = self.ffn(output) # ffn
        output = self.norm2(output_ + output)
        return output
    
    def cm_attention(self, q_e, k_w, v_w):
        '''
        q_e evidence as key
        k_w, v_w word as key and value
        '''
        # Q, K ,V
        _, _, _, d_tensor = q_e.size()
        scores = torch.matmul(q_e, k_w.transpose(2, 3)) / math.sqrt(d_tensor)
        p_attn = F.softmax(scores, dim = -1)
        out = torch.matmul(p_attn, v_w)
        # out_dim (batch, seq_len, d_model)
        return out
    
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor
    
    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
class VideoSpan(nn.Module):
    def __init__(self, configs):
        super(VideoSpan, self).__init__()
        self.ffn = nn.Linear(configs.d_model, 1)

    def forward(self, encoded_input):
        return self.ffn(encoded_input).squeeze(-1)
class OrderGenerate(nn.Module):
    def __init__(self, configs):
        super(OrderGenerate, self).__init__()
        self.ffn = nn.Linear(configs.d_model, configs.q_num)

    def forward(self, encoded_input):
        return self.ffn(encoded_input)

class LocationRegression(nn.Module):
    def __init__(self, configs):
        super(LocationRegression, self).__init__()
        self.s_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.e_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.s_index_reg = nn.Linear(configs.d_model, 1)
        self.e_index_reg = nn.Linear(configs.d_model, 1)
        self.t2v_decoder = T2Vdecoder(configs)

    def forward(self, t2v_out, v2t_out):
        encoded_input = self.t2v_decoder(t2v_out, v2t_out)
        start_logits = self.s_ffn(encoded_input)
        end_logits = self.e_ffn(start_logits)
        start_reg = torch.sigmoid(self.s_index_reg(start_logits).squeeze(-1))
        end_reg = torch.sigmoid(self.e_index_reg(end_logits).squeeze(-1))
        return start_reg, end_reg

class IoURegression(nn.Module):
    def __init__(self, configs):
        super(IoURegression, self).__init__()
        self.iou_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model),
        )
        self.iou_reg = nn.Linear(configs.d_model, 1)
        self.t2v_decoder = T2Vdecoder(configs)

    def forward(self, t2v_out, v2t_out):
        encoded_input = self.t2v_decoder(t2v_out, v2t_out)
        ious = self.iou_ffn(encoded_input)
        ious = torch.sigmoid(self.iou_reg(ious).squeeze(-1))
        return ious