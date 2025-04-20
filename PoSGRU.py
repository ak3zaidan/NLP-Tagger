from torch import nn

class PoSGRU(nn.Module) :
    def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True, embed_init=None):
        super().__init__()
        
        if embed_init is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embed_init,
                padding_idx=1, 
                freeze=False,
                sparse=False
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
            
        self.embed_to_hidden = nn.Linear(embed_dim, hidden_dim)
        self.num_layers = num_layers
        self.gru_layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=hidden_dim,
                    hidden_size=hidden_dim // 2,
                    batch_first=True,
                    bidirectional=True
                )
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.residual = residual
        
    def forward(self, x):
        embedded = self.embedding(x)
        hidden = self.embed_to_hidden(embedded)
        
        for i in range(self.num_layers):
            gru_out, _ = self.gru_layers[i](hidden)
            if self.residual:
                hidden = hidden + gru_out
            else:
                hidden = gru_out
                
        output = self.classifier(hidden)
        return output
    


    # Modified this function for Task 10

    # def forward(self, x):
    #     embedded = self.embedding(x)
        
    #     if self.num_layers == 0:
    #         hidden = self.linear_layers(embedded)
    #     else:
    #         hidden = self.embed_to_hidden(embedded)
    #         for i in range(self.num_layers):
    #             gru_out, _ = self.gru_layers[i](hidden)
    #             hidden = hidden + gru_out if self.residual else gru_out
        
    #     output = self.classifier(hidden)
    #     return output

    # Modified this function for Task 10

    # def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True, embed_init=None):
    #     super().__init__()

    #     self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)

    #     if embed_init is not None:
    #         self.embedding.weight.data = embed_init

    #     self.num_layers = num_layers
    #     self.residual = residual

    #     if num_layers == 0:
    #         # just feedforward layers on top of embeddings
    #         self.linear_layers = nn.Sequential(
    #             nn.Linear(embed_dim, hidden_dim),
    #             nn.GELU(),
    #             nn.Linear(hidden_dim, hidden_dim)
    #         )
    #     else:
    #         #linear layer to project embedding to hidden size
    #         self.embed_to_hidden = nn.Linear(embed_dim, hidden_dim)

    #         #  multi layer bi-directional Gru
    #         self.gru_layers = nn.ModuleList()
    #         for i in range(num_layers):
    #             self.gru_layers.append(
    #                 nn.GRU(
    #                     input_size=hidden_dim,
    #                     hidden_size=hidden_dim // 2,
    #                     batch_first=True,
    #                     bidirectional=True
    #                 )
    #             )

    #     self.classifier = nn.Sequential(
    #         nn.Linear(hidden_dim, hidden_dim),
    #         nn.GELU(),
    #         nn.Linear(hidden_dim, output_dim)
    #     )