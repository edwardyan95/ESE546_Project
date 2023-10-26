import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=100, embed_dim=100, num_layers=2, num_classes=119, dropout_prob=0.5):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        
        self.embed = nn.Embedding(num_embeddings=input_dim, embedding_dim=embed_dim)
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob)
        
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # Passing the input through the LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Only take the output from the final timestep
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through the fully connected layers
        output = self.fc(lstm_out)
        
        return output
