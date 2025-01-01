import torch
import numpy as np
import torch.nn as nn
from .SpectrumEmbedding import SpectrumEmbedding
from .PeptideEmbedding import PeptideEmbedding


def custom_model_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=1)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)  # 将权重初始化为1
        nn.init.zeros_(m.bias)


class DIAClip(nn.Module):
    def __init__(
            self,
            d_model = 512,
            RT_dim = 50,
            n_head = 32,
            dropout = 0.2,
            dim_feedforward = 2048,
            bin_size = 2500,
            max_mz_range = 2500,
            device = 'cuda',
            peptide_vocab = "./peptide_vocab.txt",
            modification_vocab = "./modification_vocab.txt",
            peptide_model = 'trans',
            spec_model_ratio = 0.5
    ):
        super().__init__()

        self.device = device
        self.peptide_embedding = PeptideEmbedding(
            d_model=d_model,
            device=device,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            peptide_vocab=peptide_vocab,
            modification_vocab=modification_vocab,
            num_layer=4,
            model=peptide_model
        )

        self.pre_spectrum_embedding = SpectrumEmbedding(
            d_model=d_model,
            device=device,
            n_head=n_head,
            RT_dim=RT_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bin_size=bin_size,
            max_mz_range=max_mz_range,
            num_layer=8,
            chrom_ratio=spec_model_ratio
        )

        self.frag_spectrum_embedding = SpectrumEmbedding(
            d_model=d_model,
            device=device,
            n_head=n_head,
            RT_dim=RT_dim,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bin_size=bin_size,
            max_mz_range=max_mz_range,
            num_layer=8,
            chrom_ratio=spec_model_ratio
        )

        self.ts_encoding = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout, dim_feedforward=dim_feedforward), num_layers=4)

    def pearson_correlation(self, x, y):
        """
        :param x: [N, dim]
        :param y: [N, dim]
        :return: [N]
        """
        mean_x = torch.mean(x, dim=-1)
        mean_y = torch.mean(y, dim=-1)

        # 计算分子和分母
        numerator = torch.sum((x - mean_x.unsqueeze(-1)) * (y - mean_y.unsqueeze(-1)), dim=-1)
        denominator = torch.sqrt(torch.sum((x - mean_x.unsqueeze(-1)) ** 2, dim=-1) * torch.sum((y - mean_y.unsqueeze(-1)) ** 2, dim=-1))

        return numerator / (denominator + 1e-6)

    def forward(self, data):
        peptide_feature = self.peptide_forward(data=data)
        spec_feature = self.spec_forward(data=data)

        peptide_feature = peptide_feature.squeeze(1)
        spec_feature = spec_feature.squeeze(1)
        return peptide_feature, spec_feature

    def load(self, file_path):
        """
        :param file_path: (str): 模型参数文件的路径
        """
        try:
            state_dict = torch.load(file_path, map_location=self.device, weights_only=True)  # 根据需要选择设备
            self.load_state_dict(state_dict)
            print(f"load model from {file_path}")
        except Exception as e:
            print(f"error when loading model: {e}")

    def parameter_init(self):
        self.apply(custom_model_init)

    def peptide_forward(self, data):
        peptides = data["peptide"]
        modifications = data["modification"]
        return self.peptide_embedding(peptides, modifications)

    def spec_forward(self, data):
        peptide_chrom = data["peptide_chrom"]
        peptide_mz = data["peptide_mz"]
        peptide_RT = data["peptide_RT"]
        frag_chrom = data["fragment_chrom"]
        frag_mz = data["fragment_mz"]
        frag_RT = data["fragment_RT"]

        precursor_feature = self.pre_spectrum_embedding(spec=peptide_chrom, RT=peptide_RT, mz=peptide_mz)
        fragment_feature = self.frag_spectrum_embedding(spec=frag_chrom, RT=frag_RT, mz=frag_mz)
        feature = self.ts_encoding(torch.cat([precursor_feature, fragment_feature], dim=1))
        feature = torch.mean(feature, dim=1, keepdim=True)
        return feature

if __name__ == "__main__":
    spec = torch.randn(64, 10, 100)
    mz = torch.rand(64, 10) * 5000
    RT = torch.arange(0, 100, 1).unsqueeze(0).repeat(64, 1)
    peptide = ["L N P W Y H F L M Q V A P P K", 'N L E V G R L L N I S M T M D S P K']
    modifications = ["11:Oxidation[M]", "16:Oxidation[M]"]




