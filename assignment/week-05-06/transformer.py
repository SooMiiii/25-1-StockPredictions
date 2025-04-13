import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # 위치 인코딩 미리 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 버퍼로 등록 (모델 파라미터로 취급되지 않음)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        return x + self.pe[:, :seq_len, :]


class Masking(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x.shape = (batch_size, seq_len, data_dim)
        _, seq_len, _ = x.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        
        # d_k와 d_v는 d_model을 num_heads로 나눈 값으로 설정
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 헤드당 차원
        self.d_v = d_model // num_heads  # 헤드당 차원
        
        # 선형 투영을 위한 가중치 행렬
        self.W_q = nn.Linear(d_model, d_model)  # (d_model -> d_model)
        self.W_k = nn.Linear(d_model, d_model)  # (d_model -> d_model)
        self.W_v = nn.Linear(d_model, d_model)  # (d_model -> d_model)
        
        # 출력 투영을 위한 가중치 행렬
        self.W_o = nn.Linear(d_model, d_model)  # (d_model -> d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)  # 입력의 배치 크기

        # 선형 투영: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        Q = self.W_q(query)  # Query 변환
        K = self.W_k(key)    # Key 변환
        V = self.W_v(value)  # Value 변환

        # 헤드로 분할: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # 스케일드 닷-프로덕트 어텐션 계산:
        # Q와 K의 내적을 계산하여 어텐션 스코어를 얻음
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 마스킹 적용: 특정 위치를 무시하기 위해 큰 음수 값(-1e9)을 더함
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 어텐션 가중치 계산: 소프트맥스를 통해 확률 분포로 변환
        attn_weights = torch.softmax(scores, dim=-1)

        # 어텐션 출력 계산: 가중치를 Value에 곱함
        # (batch_size, num_heads, seq_len_q, d_v)
        attn_output = torch.matmul(attn_weights, V)

        # 헤드 연결: (batch_size, num_heads, seq_len_q, d_v) -> (batch_size, seq_len_q/d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            # contiguous()는 비연속적인 텐서를 연속적인 메모리 배치로 변환, view를 쓸 때 같이 씀

        # 최종 선형 투영: (batch_size, seq_len_q/d_k) -> (batch_size/seq_len/d_model)
        output = self.W_o(attn_output)

        return output


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)  # Layer Normalization
        self.dropout = nn.Dropout(dropout)      # Dropout

    def forward(self, x, sublayer_output):
        # Residual Connection + Layer Normalization
        return self.layer_norm(x + self.dropout(sublayer_output))

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

        # Feed Forward Network (FFN)
        self.ffn = FeedForwardNetwork(d_model=d_model, ff_dim=ff_dim, dropout=dropout)

        # Residual Connection + Layer Normalization
        self.norm1 = ResidualLayerNorm(d_model=d_model, dropout=dropout)
        self.norm2 = ResidualLayerNorm(d_model=d_model, dropout=dropout)

    def forward(self, x, src_mask=None):
        # Multi-Head Attention + Residual Connection + LayerNorm
        attn_output = self.mha(x, x, x, mask=src_mask)  # Self-attention: query=key=value=x
        out1 = self.norm1(x, attn_output)

        # Feed Forward Network + Residual Connection + LayerNorm
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1, ffn_output)

        return out2

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        
        # Self Attention
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        # Encoder-Decoder Attention
        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        
        # Feed Forward Network
        self.ffn = FeedForwardNetwork(d_model=d_model, ff_dim=ff_dim, dropout=dropout)

        # Normalization layers
        self.norm1 = ResidualLayerNorm(d_model=d_model, dropout=dropout)
        self.norm2 = ResidualLayerNorm(d_model=d_model, dropout=dropout)
        self.norm3 = ResidualLayerNorm(d_model=d_model, dropout=dropout)

    def forward(self, dec_input, enc_output, dec_mask=None):
        # Self-Attention + Residual Connection + LayerNorm
        self_attn_output = self.self_attention(dec_input, dec_input, dec_input, mask=dec_mask)
        out1 = self.norm1(dec_input, self_attn_output)

        # Encoder-Decoder Attention + Residual Connection + LayerNorm
        enc_dec_attn_output = self.enc_dec_attention(out1, enc_output, enc_output)
        out2 = self.norm2(out1, enc_dec_attn_output)

        # Feed Forward Network + Residual Connection + LayerNorm
        ffn_output = self.ffn(out2)
        out3 = self.norm3(out2, ffn_output)

        return out3

class Encoder(nn.Module):
    def __init__(self, input_dim=3, num_layers=3, max_len=512, d_model=16, num_heads=4, ff_dim=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 임베딩 레이어 (여기서는 입력 차원을 모델 차원으로 변환)
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # 인코더 레이어들
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):
        # 임베딩
        x = self.embedding(x)
        
        # 위치 인코딩 적용
        x = self.positional_encoding(x)
        
        # 인코더 레이어 통과
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return x
    
class Decoder(nn.Module):
    def __init__(self, input_dim=3, num_layers=3, max_len=512, d_model=16, num_heads=4, ff_dim=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 임베딩 레이어
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        # 디코더 레이어들
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, enc_output, dec_input, dec_mask=None):
        # 임베딩
        y = self.embedding(dec_input)
        
        # 위치 인코딩 적용
        y = self.positional_encoding(y)
        
        # 디코더 레이어 통과
        for layer in self.layers:
            y = layer(dec_input=y, enc_output=enc_output, dec_mask=dec_mask)
            
        return y

class Transformer(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, num_layers=3, max_len=512, 
                 d_model=16, num_heads=4, ff_dim=32, dropout=0.1):
        super().__init__()
        
        # 인코더
        self.encoder = Encoder(
            input_dim=input_dim,
            num_layers=num_layers,
            max_len=max_len,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # 디코더
        self.decoder = Decoder(
            input_dim=input_dim,
            num_layers=num_layers,
            max_len=max_len,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )

        # 마스킹 레이어
        self.masking = Masking()
        
        # 출력 레이어
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # 마스크 생성 (디코더용)
        dec_mask = self.masking(tgt)

        # 인코더 통과
        enc_output = self.encoder(src)

        # 디코더 통과
        dec_output = self.decoder(enc_output, tgt, dec_mask)

        # 출력 레이어 통과
        output = self.output_layer(dec_output)
        
        return output