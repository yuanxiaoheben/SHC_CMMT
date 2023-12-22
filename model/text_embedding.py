import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = mask.type(torch.float32)
    return inputs + (1.0 - mask) * mask_value


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)


class WordEmbedding(nn.Module):
    def __init__(self, num_words, word_dim, drop_rate, word_vectors=None):
        super(WordEmbedding, self).__init__()
        self.is_pretrained = False if word_vectors is None else True
        if self.is_pretrained:
            self.pad_vec = nn.Parameter(torch.zeros(size=(1, word_dim), dtype=torch.float32), requires_grad=False)
            unk_vec = torch.empty(size=(1, word_dim), requires_grad=True, dtype=torch.float32)
            nn.init.xavier_uniform_(unk_vec)
            self.unk_vec = nn.Parameter(unk_vec, requires_grad=True)
            self.glove_vec = nn.Parameter(torch.tensor(word_vectors, dtype=torch.float32), requires_grad=False)
        else:
            self.word_emb = nn.Embedding(num_words, word_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, word_ids):
        if self.is_pretrained:
            word_emb = F.embedding(word_ids, torch.cat([self.pad_vec, self.unk_vec, self.glove_vec], dim=0),
                                   padding_idx=0)
        else:
            word_emb = self.word_emb(word_ids)
        return self.dropout(word_emb)


class CharacterEmbedding(nn.Module):
    def __init__(self, num_chars, char_dim, drop_rate):
        super(CharacterEmbedding, self).__init__()
        self.char_emb = nn.Embedding(num_chars, char_dim, padding_idx=0)
        kernels, channels = [1, 2, 3, 4], [10, 20, 30, 40]
        self.char_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=char_dim, out_channels=channel, kernel_size=(1, kernel), stride=(1, 1), padding=0,
                          bias=True),
                nn.ReLU()
            ) for kernel, channel in zip(kernels, channels)
        ])
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, char_ids):
        char_emb = self.char_emb(char_ids)  # (batch_size, w_seq_len, c_seq_len, char_dim)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.permute(0, 3, 1, 2)  # (batch_size, char_dim, w_seq_len, c_seq_len)
        char_outputs = []
        for conv_layer in self.char_convs:
            output = conv_layer(char_emb)
            output, _ = torch.max(output, dim=3, keepdim=False)  # reduce max (batch_size, channel, w_seq_len)
            char_outputs.append(output)
        char_output = torch.cat(char_outputs, dim=1)  # (batch_size, sum(channels), w_seq_len)
        return char_output.permute(0, 2, 1)  # (batch_size, w_seq_len, sum(channels))


class Embedding(nn.Module):
    def __init__(self, configs, bert_model=None):
        super(Embedding, self).__init__()
        self.bert = bert_model
        # output linear layer
        self.linear = nn.Linear(768, configs.d_model)
        self.linear_2 = nn.Linear(768, configs.d_model)

    def forward(self, bert_input):
        bert_out = self.bert(**bert_input)  # (batch_size, w_seq_len, word_dim)
        emb = self.linear(bert_out.last_hidden_state)  # (batch_size,  seq_len, dim)
        emb2 = self.linear_2(bert_out.pooler_output)
        return emb, emb2