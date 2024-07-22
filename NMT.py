import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import spacy
import datasets
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import tqdm
import evaluate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Encoder(nn.Module):
    def __init__(
        self, input_dim, embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, encoder_hidden_dim, bidirectional=True)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        h_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = torch.tanh(self.fc(h_cat))
        return outputs, hidden


class Attention(nn.Module):
    def __init__(
        self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v_a = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len, _, _ = encoder_outputs.shape
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v_a(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(
        self, output_dim, embedding_dim, encoder_hidden_dim,
        decoder_hidden_dim, attention, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU((encoder_hidden_dim * 2) + embedding_dim, decoder_hidden_dim,
                          bidirectional = False)
        self.attn = attention
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(encoder_hidden_dim * 2 + decoder_hidden_dim +
                            embedding_dim, output_dim)

    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(inputs.unsqueeze(0)))
        a = self.attn(hidden, encoder_outputs).unsqueeze(1)
        weighted = torch.matmul(a, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, weighted), dim = 2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        assert (output == hidden).all()
        embedded, output, weighted = embedded.squeeze(0), output.squeeze(0), weighted.squeeze(0)
        out = self.fc(torch.cat((embedded, output, weighted), dim = 1))
        return out, hidden.squeeze(0), a.squeeze(1)


class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio = 0.5):
        batch_size = src.shape[1]
        tgt_len = tgt.shape[0]
        tgt_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)
        input = tgt[0, :]
        for t in range(1, tgt_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            arg_max = output.argmax(dim = -1)
            input = tgt[t] if teacher_force else arg_max
        return outputs
