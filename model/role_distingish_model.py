# @Author       : Duhongkai
# @Time         : 2024/2/1 11:55
# @Description  : 角色区分模型

from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F



class DuelRole(nn.Module):
    def __init__(self, num_layers=22, token_dim=256, prefix_length=50, bert_hidden_size=768):
        super(DuelRole, self).__init__()
        lstm_hidden_size = 512

        self.fc_bert = nn.Linear(bert_hidden_size, lstm_hidden_size * 2)
        self.d_dd = nn.Linear(lstm_hidden_size * 2, lstm_hidden_size * 2, bias=False)
        # self.d_lstm = nn.LSTM(bert_hidden_size, lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)

        # self.p_lstm = nn.LSTM(bert_hidden_size, lstm_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.p_dd = nn.Linear(lstm_hidden_size * 2, lstm_hidden_size * 2, bias=False)
        self.fc_final = torch.nn.Linear(lstm_hidden_size * 2, num_layers * token_dim * prefix_length, bias=False)


    def forward(self, d_pooler_output, p_pooler_output):
        # doctor
        # d_lstm_output, _ = self.d_lstm(d_pooler_output)  # (bs, seq_len, lstm_hidden_size * 2)
        d_cls = self.fc_bert(d_pooler_output[:, 0, :])  # (bs, seq_len, lstm_hidden_size * 2)
        g1 = self.d_dd(d_cls)
        d_cls = g1 * d_cls + g1
        g2 = self.d_dd(d_cls)
        d_cls = g2 * d_cls + g2  # (bs, lstm_hidden_size * 2)
        # d_cat = torch.cat([d_cls, d_lstm_output[:, -1, :]], dim=1)  # (bs, lstm_hidden_size * 4)

        # patient
        # p_lstm_output, _ = self.p_lstm(p_pooler_output)
        p_cls = self.fc_bert(p_pooler_output[:, 0, :])
        d1 = self.p_dd(p_cls)
        p_cls = d1 * p_cls + d1
        d2 = self.p_dd(p_cls)
        p_cls = d2 * p_cls + d2
        # p_cat = torch.cat([p_cls, p_lstm_output[:, -1, :]], dim=1)

        # merge
        # merge = torch.stack([d_cat, p_cat], dim=1)
        merge = torch.stack([d_cls, p_cls], dim=1)
        merge = self.fc_final(merge)
        return merge

    def loss_preds(self, preds, true):
        preds = -F.log_softmax(preds, dim=1)
        loss = torch.mean(torch.sum(preds * true, dim=1))
        return loss