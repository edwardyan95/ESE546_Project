import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class RNNClassifier(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=100, embed_dim=100, num_layers=2, num_classes=119, dropout_prob=0.5, nn_type = 'LSTM'):
        super(RNNClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.nn_type = nn_type
        self.drop = nn.Dropout(dropout_prob)
        self.embed = nn.Linear(input_dim, embed_dim)
        if nn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        elif nn_type == 'GRU':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        print('Num parameters: ', sum([p.numel() for p in self.parameters()]))
        
    def forward(self, x):
        x = self.drop(self.embed(x))
        #print('x shape:', x.shape)
        # Passing the input through the LSTM layers
        rnn_out, _ = self.rnn(x)
        #print('out shape:', rnn_out.shape)
        rnn_out = self.drop(rnn_out)
        rnn_out = torch.add(rnn_out, x) # residual
        # Only take the output from the final timestep
        rnn_out = rnn_out[:, -1, :]
        #print('rnn out shape:', rnn_out.shape)
        # Pass through the fully connected layers
        output = self.fc(rnn_out)
        #print('fc output:', output.shape)
        # return F.log_softmax(output)
        return output
    
    # def init_hidden(self, batch_size):
    #     weight = next(self.parameters()).data
    #     hidden = (weight.new(self.num_layers, batch_size, self.hidden_dim).zero_(),
    #                   weight.new(self.num_layers, batch_size, self.hidden_dim).zero_())
    #     return hidden

class WarmupWithScheduledDropLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, initial_lr, drop_epochs, drop_factor=0.5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.drop_epochs = drop_epochs
        self.drop_factor = drop_factor
        super(WarmupWithScheduledDropLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.initial_lr * alpha for base_lr in self.base_lrs]
        else:
            # Scheduled drop
            lr = self.initial_lr
            for epoch in self.drop_epochs:
                if self.last_epoch >= epoch:
                    lr *= self.drop_factor
            return [lr for base_lr in self.base_lrs]

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        seq_len: the length of the incoming sequence (default=50).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, seq_len=30):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = torch.transpose(x, 0, 1) # accommondate batch_first = True
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return torch.transpose(x, 0, 1)

class TransformerClassifier(nn.Transformer):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, input_dim, hidden_dim, nlayers, nhead, num_classes, dropout=0.5):
        super(TransformerClassifier, self).__init__(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, num_encoder_layers=nlayers, batch_first=True)
        self.model_type = 'Transformer'
        self.src_mask = None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_classes)
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embed.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=False):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None

        src = self.embed(src) * math.sqrt(self.hidden_dim)
        src = torch.cat((self.class_token.expand(src.shape[0], 1, -1), src), dim=1)
        pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, seq_len=src.shape[1]+1) # src shape: n_batch x n_timestep (add class token) x hidden_dim
        src = pos_encoder(src)
        output = self.encoder(src, mask=self.src_mask)
        #print('encoder output: ', output.shape)
        output = output[:,0,:]
        output = self.decoder(output)
        #print('decoder output: ', output.shape)
        return output