# =============================================================================
# File: transformer_model.py
# Purpose: Temporal modeling architectures for latent sequence conditioning (vanilla, bidirectional, cross-attention, gated variants).
# Prerequisites: Imported by train/test scripts that enable --use_transformer.
# Usually Run Next: Selected by CLI flags in train_VAE_SYNBADD_svf.py and test_model_subjectwise.py.
# Model Components Used: Latest ablation workflows use CrossAttentionTransformer, CrossAttentionGatedTransformer, and GatedCrossAttentionTransformer variants.
# =============================================================================

import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5):
        super(TransformerModel, self).__init__()
        
        # Positional embeddings (learnable)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            num_layers=num_layers,
            encoder_layer=encoder_layer,)
        
        # Average pooling over temporal dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Assuming x is of shape (batch_size, seq_len,  input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Add positional embeddings (broadcast across batch)
        positional_embeddings = self.positional_embeddings[:, :seq_len, :]
        x = x + positional_embeddings
        
        # Pass through the transformer encoder
        # transformer expects (seq_len, batch, input_dim)
        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)
        
        # Apply average pooling over the temporal dimension
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        x = x.view(batch_size, -1)

        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5):
        super(TransformerModel, self).__init__()
        
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim

        # No CLS token needed here
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))  # Only max_seq_len

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            num_layers=num_layers,
            encoder_layer=encoder_layer,
        )

        # Attention pooling: simple FC layer to produce attention scores
        self.attention_fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Add positional embeddings
        positional_embeddings = self.positional_embeddings[:, :seq_len, :]
        x = x + positional_embeddings

        # Transformer expects (seq_len, batch, input_dim)
        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)  # back to (batch_size, seq_len, input_dim)

        # Attention pooling over sequence dimension
        attn_scores = self.attention_fc(x)  # (batch_size, seq_len, 1)
        attn_scores = attn_scores.squeeze(-1)  # (batch_size, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1)  # softmax over time

        # Weighted sum of the timepoint features
        pooled_output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)

        return pooled_output
    
class GatedTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5):
        super(GatedTransformerModel, self).__init__()

        self.max_seq_len = max_seq_len
        self.input_dim = input_dim

        # Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Gated attention heads
        self.attention_fc = nn.Linear(input_dim, 1)  # For attention scores
        self.gating_fc = nn.Linear(input_dim, 1)     # For gating scores (sigmoid)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Add positional embeddings
        positional_embeddings = self.positional_embeddings[:, :seq_len, :]
        x = x + positional_embeddings

        # Transformer expects (seq_len, batch, input_dim)
        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)  # back to (batch_size, seq_len, input_dim)

        # Gated Attention Pooling
        attn_scores = self.attention_fc(x).squeeze(-1)  # (batch_size, seq_len)
        gating_scores = torch.sigmoid(self.gating_fc(x)).squeeze(-1)  # (batch_size, seq_len)

        # Combine attention and gating
        gated_attn_scores = attn_scores * gating_scores  # (batch_size, seq_len)

        attn_weights = torch.softmax(gated_attn_scores, dim=1)  # softmax over time

        # Weighted sum of timepoint features
        pooled_output = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)

        return pooled_output

import torch
import torch.nn as nn
import math

class GatedCrossAttentionTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5, use_positional_embeddings=True):
        super(GatedCrossAttentionTransformer, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.use_positional_embeddings = use_positional_embeddings

        # Positional embeddings
        if use_positional_embeddings:
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
            nn.init.xavier_uniform_(self.positional_embeddings)
        else:
            self.register_buffer('positional_embeddings', self._generate_sinusoidal_embeddings(max_seq_len, input_dim))

        # Timepoint embedding for missing token conditioning
        self.missing_time_embedding = nn.Embedding(max_seq_len, input_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Gated attention scoring
        self.attention_fc = nn.Linear(input_dim, 1)
        self.gating_fc = nn.Linear(input_dim, 1)

    def _generate_sinusoidal_embeddings(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x, missing_token_index):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Positional encoding
        x = x + self.positional_embeddings[:, :seq_len, :]

        # Transformer encoder
        x = x.transpose(0, 1)  # (seq_len, batch_size, input_dim)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, input_dim)

        # Gated Attention
        attn_scores = self.attention_fc(x).squeeze(-1)  # (batch_size, seq_len)
        gating_scores = torch.sigmoid(self.gating_fc(x)).squeeze(-1)  # (batch_size, seq_len)
        gated_scores = attn_scores * gating_scores  # (batch_size, seq_len)
        gated_attn_weights = torch.softmax(gated_scores, dim=-1)  # (batch_size, seq_len)

        # Apply gating to sequence
        x_gated = gated_attn_weights.unsqueeze(-1) * x  # (batch_size, seq_len, input_dim)

        # Get query vector for cross attention
        query = self.missing_time_embedding(missing_token_index.squeeze(-1))  # (batch_size, input_dim)

        # Cross attention: query over gated sequence
        attn_scores_cross = (x_gated * query.unsqueeze(1)).sum(dim=-1)  # (batch_size, seq_len)
        attn_weights_cross = torch.softmax(attn_scores_cross, dim=-1)   # (batch_size, seq_len)

        # Final attention pooling
        output = torch.bmm(attn_weights_cross.unsqueeze(1), x_gated).squeeze(1)  # (batch_size, input_dim)

        return output

class CrossAttentionTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5, use_positional_embeddings=True):
        super(CrossAttentionTransformer, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.use_positional_embeddings = use_positional_embeddings

        # Positional embeddings
        if use_positional_embeddings:
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
            nn.init.xavier_uniform_(self.positional_embeddings)
        else:
            self.register_buffer('positional_embeddings', self._generate_sinusoidal_embeddings(max_seq_len, input_dim))

        # Timepoint embedding for missing token conditioning
        self.missing_time_embedding = nn.Embedding(max_seq_len, input_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Attention scoring (no gating)
        self.attention_fc = nn.Linear(input_dim, 1)

    def _generate_sinusoidal_embeddings(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x, missing_token_index):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Positional encoding
        x = x + self.positional_embeddings[:, :seq_len, :]

        # Transformer encoder
        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)

        # Attention (subject-specific selection without gating)
        attn_scores = self.attention_fc(x).squeeze(-1)  # (batch_size, seq_len)

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch_size, seq_len)

        # Sequence features with attention (no gating)
        x_attended = torch.bmm(attn_weights.unsqueeze(1), x)  # (batch_size, 1, input_dim)
        x_attended = x_attended.squeeze(1)  # (batch_size, input_dim)

        # Get missing time embedding (query)
        missing_time_embed = self.missing_time_embedding(missing_token_index.squeeze(-1))  # (batch_size, input_dim)

        # Cross attention: Query missing_time_embed over attended x
        attn_scores_cross = (missing_time_embed.unsqueeze(1) * x).sum(dim=-1)  # (batch_size, seq_len)
        attn_weights_cross = torch.softmax(attn_scores_cross, dim=-1)  # (batch_size, seq_len)

        # Weighted sum to get final pooled output
        output = torch.bmm(attn_weights_cross.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)

        return output
    
class CrossAttentionGatedTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5, use_positional_embeddings=True):
        super(CrossAttentionGatedTransformer, self).__init__()

        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.use_positional_embeddings = use_positional_embeddings

        # Positional embeddings
        if use_positional_embeddings:
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
            nn.init.xavier_uniform_(self.positional_embeddings)
        else:
            self.register_buffer('positional_embeddings', self._generate_sinusoidal_embeddings(max_seq_len, input_dim))

        # Timepoint embedding for missing token conditioning
        self.missing_time_embedding = nn.Embedding(max_seq_len, input_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        # Cross attention scoring
        self.attention_fc = nn.Linear(input_dim, 1)
        
        # Gating mechanism (applied after cross attention)
        self.gating_fc = nn.Linear(input_dim, 1)

    def _generate_sinusoidal_embeddings(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x, missing_token_index):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Positional encoding
        x = x + self.positional_embeddings[:, :seq_len, :]

        # Transformer encoder
        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)

        # Get missing time embedding (query)
        missing_time_embed = self.missing_time_embedding(missing_token_index.squeeze(-1))  # (batch_size, input_dim)

        # Cross attention first: Query missing_time_embed over x
        attn_scores_cross = (missing_time_embed.unsqueeze(1) * x).sum(dim=-1)  # (batch_size, seq_len)
        attn_weights_cross = torch.softmax(attn_scores_cross, dim=-1)  # (batch_size, seq_len)
        
        # Get cross-attended features
        x_cross = torch.bmm(attn_weights_cross.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)
        
        # Expand x_cross to match original sequence dimension for gating
        x_cross_expanded = x_cross.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, input_dim)
        
        # Compute attention scores based on cross-attended features
        attn_scores = self.attention_fc(x).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply gating after cross attention
        gating_scores = torch.sigmoid(self.gating_fc(x_cross_expanded)).squeeze(-1)  # (batch_size, seq_len)
        
        # Combine attention with gating
        gated_scores = attn_scores * gating_scores  # (batch_size, seq_len)
        
        # Apply softmax to get final attention weights
        final_attn_weights = torch.softmax(gated_scores, dim=-1)  # (batch_size, seq_len)
        
        # Weighted sum to get final pooled output
        output = torch.bmm(final_attn_weights.unsqueeze(1), x).squeeze(1)  # (batch_size, input_dim)

        return output

class TransformerBidirectional(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, max_seq_len=5, use_positional_embeddings=True, use_bidirectional_embeddings=True):
        super(TransformerBidirectional, self).__init__()
        
        # Positional embeddings (learnable)
        if use_positional_embeddings:
            self.positional_embeddings = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
            nn.init.xavier_uniform_(self.positional_embeddings)
        else: # use sinusoidal positional embeddings
            # Initialize sinusoidal embeddings
            self.register_buffer('positional_embeddings', self._generate_sinusoidal_embeddings(max_seq_len, input_dim))



        # bidirectional learnable embeddings
        self.past_embeddings = nn.Parameter(torch.zeros(1, input_dim))
        self.future_embeddings = nn.Parameter(torch.zeros(1, input_dim))
        self.missing_token_embeddings = nn.Parameter(torch.zeros(1, input_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer_model = nn.TransformerEncoder(
            num_layers=num_layers,
            encoder_layer=encoder_layer,)
        
        # Average pooling over temporal dimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.use_positional_embeddings = use_positional_embeddings
        self.use_bidirectional_embeddings = use_bidirectional_embeddings

    def _generate_sinusoidal_embeddings(self, max_len, d_model):
        """Generates sinusoidal embeddings as a buffer."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        return pe

    def forward(self, x, missing_token_index):
        # x: (batch_size, seq_len, input_dim)
        # missing_token_index: (batch_size, 1)
        batch_size, seq_len, input_dim = x.size()
        
        # Create time indices (1D tensor: [0, 1, ..., seq_len-1])
        time = torch.arange(seq_len, device=x.device).unsqueeze(0)  # shape (1, seq_len)
        
        # Expand missing_token_index to (batch_size, seq_len)
        missing_index_expanded = missing_token_index.expand(-1, seq_len)  # shape (batch_size, seq_len)

        # Create masks
        past_mask = time < missing_index_expanded
        future_mask = time > missing_index_expanded
        missing_mask = time == missing_index_expanded

        # Initialize embedding_token
        embedding_token = torch.zeros_like(x)

        # Broadcast embeddings using masks
        embedding_token += past_mask.unsqueeze(2) * self.past_embeddings  # shape (1, input_dim)
        embedding_token += future_mask.unsqueeze(2) * self.future_embeddings
        embedding_token += missing_mask.unsqueeze(2) * self.missing_token_embeddings

        # Add token and positional embeddings
        if self.use_bidirectional_embeddings:
            x = x + embedding_token
        x = x + self.positional_embeddings[:, :seq_len, :]

        # Transformer expects (seq_len, batch, input_dim)
        x = x.transpose(0, 1)
        x = self.transformer_model(x)
        x = x.transpose(0, 1)

        # Gather output at each sample's missing_token_index
        output = x.gather(1, missing_token_index.unsqueeze(2).expand(-1, -1, input_dim)).squeeze(1)

        return output
