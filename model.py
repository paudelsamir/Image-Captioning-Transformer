import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_length, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        # Embedding layer for input captions
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Linear layer to project image features
        self.image_projection = nn.Linear(512, embed_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_dim))
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # Important: use batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final linear layer to predict vocabulary tokens
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, tgt, memory, tgt_mask=None):
        # Project image features to the correct dimension
        memory = self.image_projection(memory).unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Create attention mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        # Embed the target sequence and add positional encoding
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        tgt_emb = self.dropout(tgt_emb)
        
        # Pass through transformer decoder
        output = self.transformer_decoder(
            tgt=tgt_emb,  # [batch_size, seq_len, embed_dim]
            memory=memory,  # [batch_size, 1, embed_dim]
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary size
        output = self.fc_out(output)  # [batch_size, seq_len, vocab_size]
        return output
