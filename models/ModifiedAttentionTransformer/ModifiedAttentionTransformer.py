import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise_linear = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.output_linear = nn.Linear(hidden_dim, dim)  

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2) 
        x = self.activation(x)
        x = self.pointwise_linear(x)
        x = self.activation(x)
        x = self.output_linear(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_taa):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        if use_taa:
            self.W_q = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
            self.W_k = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
            self.W_v = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
            self.W_o = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
        else:
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, use_taa, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        if use_taa:
            gelu = nn.GELU()
            attn_probs = gelu(attn_scores)
        else:
            attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x, use_taa):
        if use_taa:
            batch_size, seq_length, d_model = x[0].size()
            return x[0].view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        else:
            batch_size, seq_length, d_model = x.size()
            return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, use_taa, mask=None):
        Q = self.split_heads(self.W_q(Q), use_taa)
        K = self.split_heads(self.W_k(K), use_taa)
        V = self.split_heads(self.W_v(V), use_taa)

        attn_output = self.scaled_dot_product_attention(Q, K, V, use_taa, mask)
        if use_taa:
            output, _ = self.W_o(self.combine_heads(attn_output))
        else:
            output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


### Encoder without MixFFN
class EncoderLayerWithoutMixFFN(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, d_ff, dropout, use_taa):
        super(EncoderLayerWithoutMixFFN, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_taa)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_taa = use_taa

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, self.use_taa, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


### Encoder with MixFFN
class EncoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, d_ff, dropout, use_taa):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_taa)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MixFFN(d_model, hidden_dim)
        self.use_taa = use_taa

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, self.use_taa, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Mix-FFN
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x


### Decoder without MixFFN
class DecoderLayer(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, d_ff, dropout, use_taa):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_taa)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, use_taa)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_taa = use_taa

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, self.use_taa, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, self.use_taa, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


### Decoder with MixFFN
class DecoderLayerWithMixFFN(nn.Module):
    def __init__(self, d_model, hidden_dim, num_heads, d_ff, dropout, use_taa):
        super(DecoderLayerWithMixFFN, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_taa)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, use_taa)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = MixFFN(d_model, hidden_dim)
        self.use_taa = use_taa

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, self.use_taa, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, self.use_taa, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        # Mix-FFN
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)
        return x


class TransformerForSignalEstimation(nn.Module):
    def __init__(self, config):
        super(TransformerForSignalEstimation, self).__init__()

        embedding_dim = config.get("embedding_dim", 128)
        token_size = config.get("token_size", 10)
        num_heads = config.get("num_heads", 16)
        num_layers = config.get("num_layers", 8)
        hidden_dim = config.get("hidden_dim", 512)
        d_ff = config.get("d_ff", 64)
        seq_len = config.get("seq_len", 1000)
        dropout = config.get("dropout", 0.1)
        use_positional_encoding=config.get("use_positional_encoding", False)
        use_features=config.get("use_features", False)
        use_MixFFN=config.get("use_MixFFN", True)
        use_lstm_embedding=config.get("use_lstm_embedding", True)
        use_taa=config.get("use_time_aware_attention", False)

        self.use_features = use_features
        self.use_positional_encoding = use_positional_encoding
        self.use_lstm_embeddings = use_lstm_embedding
        self.use_taa = use_taa
        self.seq_len = seq_len 
        self.embedding_dim = embedding_dim
        self.token_size = token_size 
        self.no_tokens = seq_len // token_size 

        if self.use_lstm_embeddings:
            self.lstm = nn.LSTM(input_size=self.token_size, hidden_size=self.embedding_dim, batch_first=True)
        else:
            self.encoder_input_layer = nn.Linear(self.token_size, embedding_dim)
            

        if self.use_positional_encoding:
            if self.use_features:
                self.positional_encoding = PositionalEncoding(embedding_dim*2, seq_len)
            else:
                self.positional_encoding = PositionalEncoding(embedding_dim, seq_len)


        if use_features:
            self.features_linear_1 = nn.Linear(26, seq_len)
            self.features_linear_2 = nn.Linear(self.token_size, embedding_dim)
            
            if self.use_positional_encoding:
                if use_MixFFN:
                    self.encoder_layers = nn.ModuleList(
                        [EncoderLayer(embedding_dim*2, hidden_dim, num_heads, d_ff, dropout, self.use_taa) for _ in range(num_layers)]
                    )
                else:
                    self.encoder_layers = nn.ModuleList(
                        [EncoderLayerWithoutMixFFN(embedding_dim*2, hidden_dim, num_heads, d_ff, dropout, self.use_taa) for _ in range(num_layers)]
                    )
            else:
                self.encoder_layers = nn.ModuleList(
                    [EncoderLayer(embedding_dim*2, hidden_dim, num_heads, d_ff, dropout, self.use_taa) for _ in range(num_layers)]
                )
            self.output_layer = nn.Linear(embedding_dim*2, self.token_size)
            self.output_layer_2 = nn.Linear(self.seq_len, self.seq_len)
        else:
            if self.use_positional_encoding:
                if use_MixFFN:
                    self.encoder_layers = nn.ModuleList(
                        [EncoderLayer(embedding_dim, hidden_dim, num_heads, d_ff, dropout, self.use_taa) for _ in range(num_layers)]
                    )
                else:
                    self.encoder_layers = nn.ModuleList(
                        [EncoderLayerWithoutMixFFN(embedding_dim, hidden_dim, num_heads, d_ff, dropout, self.use_taa) for _ in range(num_layers)]
                    )
            else:
                self.encoder_layers = nn.ModuleList(
                    [EncoderLayer(embedding_dim, hidden_dim, num_heads, d_ff, dropout, self.use_taa) for _ in range(num_layers)]
                )
            # Output layer to map back to signal space
            self.output_layer = nn.Linear(embedding_dim, self.token_size)
            self.output_layer_2 = nn.Linear(self.seq_len, self.seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, features):
        batch_size, seq_len = x.shape
        
        if self.use_lstm_embeddings:
            x = x.view(batch_size, self.no_tokens, self.token_size)
            x = x.transpose(1, 2)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            x = x.view(batch_size, self.no_tokens, self.embedding_dim)
            
        else:
            x = x.view(batch_size, self.no_tokens, self.token_size)
            x = self.encoder_input_layer(x)
        
        if self.use_features:
            features = self.features_linear_1(features)
            if self.use_lstm_embeddings:
                features = features.view(batch_size, self.no_tokens, self.token_size)
                features = self.features_linear_2(features)
            else:
                features = features.view(batch_size, self.no_tokens, self.token_size)
                features = self.features_linear_2(features)
            x = torch.dstack((x, features))

        if self.use_positional_encoding:
            src_encoded = self.dropout(self.positional_encoding(x))
        else:
            src_encoded = self.dropout(x)
        
        # Pass through encoder layers
        enc_output = src_encoded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, mask=None) 

        # Map to output signal
        output = self.output_layer(enc_output)

        output = output.view(batch_size, seq_len)
        output = self.output_layer_2(output)
        return output