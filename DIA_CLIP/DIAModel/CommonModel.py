import torch
import torch.nn as nn

class PositionEmbedding():
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def embedding(self, shape, position=None):
        """
        :param shape: (batch, seq, d_model)
        :return:
        """
        if position is not None:
            B, seq, d_model = shape
            div_term = torch.exp(torch.arange(0, d_model, 2) * -torch.log(torch.tensor(10000.0)) / d_model).to(self.device)  # [d_model]
            pe = torch.zeros(B, seq, d_model).to(self.device)
            pe[:, :, 0::2] = torch.sin(position * div_term)  # 偶数维度
            pe[:, :, 1::2] = torch.cos(position * div_term)  # 奇数维度
            return pe
        else:
            B, seq, d_model = shape
            position = torch.arange(0, seq).unsqueeze(-1).to(self.device)  # [seq, 1]
            div_term = torch.exp(torch.arange(0, d_model, 2) * -torch.log(torch.tensor(10000.0)) / d_model).to(self.device)  # [d_model/2]
            pe = torch.zeros(seq, d_model).to(self.device)
            pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
            pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
            pe = pe.unsqueeze(0).repeat(B, 1, 1)
        return pe

class MaskTransformer(nn.Module):
    def __init__(self, d_model=512, n_head=64, dim_feedforward=2048, dropout=0.25):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.multiheadAttention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=True)
        self.feedforward_layer = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm_layer = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask):
        """
        :param x: [batch, seq, d_model]
        :param attn_mask: [batch, seq, seq]
        :return: [batch, seq, d_model]
        """
        B, seq, d_model = x.shape
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)
        mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1).reshape(B * self.n_head, seq, seq)
        attn_output, attn_weight = self.multiheadAttention(query, key, value, attn_mask=mask)
        x = self.norm_layer(x + attn_output)
        x = self.norm_layer(x + self.feedforward_layer(x))
        return x, attn_mask

class GaussianKernel(nn.Module):
    def __init__(self, centers, sigma):
        """
        :param centers: [bin_size]
        :param sigma: tensor(float)
        """
        super().__init__()
        self.centers = centers
        self.sigma = sigma

    def forward(self, x):
        """
        :param x: [batch, x, ion_num]
        :return feature: [batch, x, bin_size]
        """
        centers = self.centers.reshape(1,1,1,-1) #[1, 1, 1, bin_size]
        x = x.unsqueeze(-1) # [batch, x, ion_num, 1]
        dist = (x - centers) ** 2
        gaussian_features = torch.exp(-dist / (2 * self.sigma ** 2))
        gaussian_features = torch.sum(gaussian_features, dim=2, keepdim=False)
        return gaussian_features

if __name__ == "__main__":
    x = torch.randn(1, 1, 2)
    centers = torch.linspace(0, 2, 5)
    kernel = GaussianKernel(centers, 1.0)
    print(kernel(x))