import torch
import torch.nn as nn
from .CommonModel import PositionEmbedding, MaskTransformer, GaussianKernel


class SpectrumEmbedding(nn.Module):
    def __init__(
            self,
            d_model=512,
            RT_dim=32,
            bin_size=2500,
            max_mz_range=2500,
            n_head=16,
            dim_feedforward=1024,
            dropout=0.2,
            num_layer=8,
            device='cuda',
            chrom_ratio=0.5
    ):
        super().__init__()
        self.chrom_embedding = ChromatogramEmbedding(
            d_model=d_model,
            device=device,
            n_head=n_head,
            RT_dim=RT_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layer=num_layer
        )

        self.mz_bin_embedding = MZBinEmbedding(
            d_model=d_model,
            device=device,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bin_size=bin_size,
            max_mz_range=max_mz_range,
            num_layer=num_layer
        )
        self.chrom_ratio = chrom_ratio

    def forward(self, spec, mz, RT):
        """
        :param spec: [batch, ion_num, RT_dim]
        :param RT: [batch, RT_dim]
        :param mz: [batch, ion_num]
        :return: [batch, 1, d_model]
        """
        chrom_feature = self.chrom_embedding(chrom=spec, RT=RT, mz=mz)
        chrom_feature = torch.mean(chrom_feature, dim=1, keepdim=True)
        spec_feature = self.mz_bin_embedding(spec=spec, RT=RT, mz=mz)
        spec_feature = torch.mean(spec_feature, dim=1, keepdim=True)

        feature = self.chrom_ratio * chrom_feature + (1-self.chrom_ratio) * spec_feature
        return feature



class ChromatogramEmbedding(nn.Module):
    def __init__(
            self,
            d_model = 512,
            RT_dim = 32,
            n_head = 16,
            dim_feedforward = 1024,
            dropout=0.2,
            num_layer = 8,
            device='cuda'
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.RT_dim = RT_dim
        self.ts_encoding = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout, dim_feedforward=dim_feedforward), num_layers=num_layer)
        self.position_emb = PositionEmbedding(device=device)
        self.dropout = nn.Dropout(p=dropout)
        self.chrom_proj = nn.Linear(self.RT_dim, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.batch_norm = nn.BatchNorm1d(self.d_model)

    def forward(self, chrom, mz, RT):
        """
        :param spec: [batch, ion_num, RT_dim]
        :param RT: [batch, RT_dim]
        :param mz: [batch, ion_num]
        :return: [batch, ion_num + 1, d_model]
        """
        B, ion_num, RT_dim = chrom.shape
        feature = torch.cat([RT.unsqueeze(1), chrom], dim=1) #[batch, ion_num + 1, RT_dim]
        feature = self.chrom_proj(feature) #[batch, ion_num + 1, d_model]
        feature = self.dropout(feature) #[batch, ion_num + 1, d_model]
        feature = feature.reshape(B * (ion_num + 1), self.d_model)
        feature = self.batch_norm(feature)
        feature = feature.reshape(B, ion_num + 1, self.d_model)
        mz_emb = self.position_emb.embedding(shape=(B, ion_num, self.d_model), position=mz.unsqueeze(-1)) #[batch, ion_num, d_model]
        feature[:, 1:, :] += mz_emb
        feature = self.ts_encoding(feature)
        return feature

class MZBinEmbedding(nn.Module):
    def __init__(
            self,
            d_model = 512,
            bin_size=2500,
            max_mz_range=2500,
            n_head = 16,
            dim_feedforward = 1024,
            dropout=0.2,
            num_layer = 8,
            device = 'cpu'
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.bin_size = bin_size
        self.max_mz_range = max_mz_range
        self.ts_encoding = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout, dim_feedforward=dim_feedforward), num_layers=num_layer)
        self.position_emb = PositionEmbedding(device=device)
        self.mz_bin_proj = nn.Linear(self.bin_size, self.d_model)
        self.intensity_bin_proj = nn.Linear(self.bin_size, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.batch_norm = nn.BatchNorm1d(self.d_model)
        self.dropout = nn.Dropout(p=dropout)

        mz_bin_centers = torch.linspace(0.0, self.max_mz_range, self.bin_size).to(self.device)
        self.mz_gaussian_kernel = GaussianKernel(centers=mz_bin_centers, sigma=2 * self.max_mz_range/self.bin_size)
        intensity_bin_centers = torch.linspace(0.0, 1.0, self.bin_size).to(self.device)
        self.intensity_gaussian_kernel = GaussianKernel(centers=intensity_bin_centers, sigma=2 * 1.0/self.bin_size)

    def forward(self, spec, mz, RT):
        """
        :param spec: [batch, ion_num, RT_dim]
            RT: [batch, RT_dim]
            mz: [batch, ion_num]
        :return: [batch, RT_dim, d_model]
        """
        B, ion_num, RT_dim = spec.shape
        spec = spec.permute(0,2,1) # [batch, RT_dim, ion_num]
        intensity_feature = self.intensity_gaussian_kernel(spec)
        intensity_feature = self.intensity_bin_proj(intensity_feature)
        intensity_feature = self.dropout(intensity_feature)  # [batch, RT_dim, bin_size]
        intensity_feature = intensity_feature.reshape(B * RT_dim, self.d_model)
        intensity_feature = self.batch_norm(intensity_feature)
        intensity_feature = intensity_feature.reshape(B, RT_dim, self.d_model)

        mz = mz.unsqueeze(1).repeat(1, RT_dim, 1)
        mz_feature = self.mz_gaussian_kernel(mz)
        mz_feature = self.mz_bin_proj(mz_feature)
        mz_feature = self.dropout(mz_feature)  # [batch, RT_dim, bin_size]
        mz_feature = mz_feature.reshape(B * RT_dim, self.d_model)
        mz_feature = self.batch_norm(mz_feature)
        mz_feature = mz_feature.reshape(B, RT_dim, self.d_model)

        feature = intensity_feature + mz_feature
        feature += self.position_emb.embedding(shape=(B, RT_dim, self.d_model))
        feature = self.ts_encoding(feature)
        return feature