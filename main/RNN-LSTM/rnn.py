import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, rnn_type, dropout, layer_norm, device='cuda'):
        super(RNN_model, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.num_directions = 2 if bidirectional == True else 1
        self.device = device
        self.layer_norm = layer_norm
        
        # rnn_type에 따른 recurrent layer 설정
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout = dropout)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout = dropout)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout = dropout)
        
        # Define the Layer Normalization Layer
        self.layer_norm = nn.LayerNorm(hidden_size) 
        
        # bidirectional에 따른 fc layer 구축
        # bidirectional 여부에 따라 hidden state의 shape가 달라짐 (True: 2 * hidden_size, False: hidden_size)
        self.fc = nn.Linear(self.num_directions * hidden_size, 1)


    def forward(self, x):
        # initial hidden states 설정
        h0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(self.device)
        
        # 선택한 rnn_type의 RNN으로부터 output 도출
        if self.rnn_type in ['rnn', 'gru']:
            out, _ = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        else:
            # LSTM의 경우 cell state가 필요 - initial cell states 설정
            c0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(self.device)
            out, _ = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        if self.layer_norm:
            out = self.layer_norm(out)
        
        out = self.fc(out[:, -1, :])      # 마지막 seq_lengh
        # out = self.fc(torch.mean(out,1))    # 평균 사용

        return out
    

class Attention(nn.Module):
    def __init__(self, hidden_size, device):
        super(Attention, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.concat_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, rnn_outputs, final_hidden_state):
        # rnn_output.shape:         (batch_size, seq_len, hidden_size)
        # final_hidden_state.shape: (batch_size, hidden_size)
        
        batch_size, seq_len, _ = rnn_outputs.shape
        
        attn_value = self.attn(rnn_outputs) # (batch_size, seq_len, hidden_dim)
        final_hidden_state = final_hidden_state.unsqueeze(2) # (batch_size, hidden_dim, 1)
        attn_value = torch.bmm(attn_value, final_hidden_state) # (batch_size, seq_len, 1)
        attn_probability = F.softmax(attn_value.squeeze(2), dim=1) # (batch_size, seq_len)

        context = torch.bmm(rnn_outputs.transpose(1, 2), attn_probability.unsqueeze(2)).squeeze(2)
                 # (batch_size, hidden_size, seq_len) X (batch_size, seq_len, 1) = (batch_size, hidden_size)
        attn_hidden = torch.tanh(self.concat_linear(torch.cat((context, final_hidden_state.squeeze(2)), dim=1)))
        # attn_hidden = torch.tanh(context)
        return attn_hidden, attn_probability

class RNN_Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, rnn_type, dropout, layer_norm, device='cuda'):
        super(RNN_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional == True else 1
        self.rnn_type = rnn_type
        self.device = device
        self.layer_norm = layer_norm
        
        # rnn_type에 따른 recurrent layer 설정
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
        # Define the Layer Normalization Layer
        self.layer_norm = nn.LayerNorm(hidden_size) 

        # Attention module활용
        self.attn = Attention(hidden_size * self.num_directions, device)
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x):
        batch_size, _, seq_len = x.shape
        
        # data dimension: (batch_size x input_size x seq_len) -> (batch_size x seq_len x input_size)로 변환
        # x = torch.transpose(x, 1, 2)
        
        # initial hidden states 설정
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device)
        
        # 선택한 rnn_type의 RNN으로부터 output 도출
        if self.rnn_type in ['rnn', 'gru']:
            out, hidden = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        else:
            # initial cell states 설정
            c0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(self.device)
            out, hidden = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        if self.layer_norm:
            out = self.layer_norm(out)
        
        final_state = hidden.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)[-1]
        
        # Handle directions
        final_hidden_state = None
        if self.num_directions == 1:
            final_hidden_state = final_state.squeeze(0)
        elif self.num_directions == 2:
            h_1, h_2 = final_state[0], final_state[1]
            final_hidden_state = torch.cat((h_1, h_2), 1)  # Concatenate both states

        # Push through attention layer
        attn_output, attn_scores = self.attn(out, final_hidden_state)
        attn_output = self.fc(attn_output)

        return attn_output, attn_scores