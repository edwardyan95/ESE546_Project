import torch.nn as nn
import torch.nn.functional as F
import torch

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=100, embed_dim=100, num_layers=2, num_classes=119, dropout_prob=0.5):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        self.drop = nn.Dropout(dropout_prob)
        self.embed = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        print('Num parameters: ', sum([p.numel() for p in self.parameters()]))
        
    def forward(self, x):
        x = self.drop(self.embed(x))
        #print(x.shape)
        # print('emb:', emb)
        # Passing the input through the LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = self.drop(lstm_out)
        # Only take the output from the final timestep
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through the fully connected layers
        output = self.fc(lstm_out)
        
        return F.log_softmax(output)
    
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