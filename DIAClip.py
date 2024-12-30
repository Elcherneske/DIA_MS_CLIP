import torch
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import esm

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
            RT_dim = 32,
            n_head = 16,
            dropout = 0.2,
            dim_feedforward = 1024,
            bin_size = 2500,
            max_mz_range = 5000,
            device = 'cuda',
            peptide_vocab = "./peptide_vocab.txt",
            modification_vocab = "./modification_vocab.txt",
            mode_peptide = 'trans',
            mode_spec = 'spec_split_RT'
    ):
        super().__init__()

        self.mode_peptide = mode_peptide
        self.mode_spec = mode_spec
        print(f"DIA CLIP mode_peptide:{self.mode_peptide}, mode_spec:{self.mode_spec}")

        self.device = device

        if self.mode_peptide == "trans":
            self.peptide_embedding = PeptideTransformer(
                d_model=d_model,
                device=device,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                peptide_vocab=peptide_vocab,
                modification_vocab=modification_vocab,
                num_layer=4
            )
        elif self.mode_peptide == "esm2":
            self.peptide_embedding = ESMEmbdedding(d_model=d_model, device=device)

        if self.mode_spec == "chrom":
            self.chrom_embedding = ChromatogramTransformer(
                d_model=d_model,
                device=device,
                n_head=n_head,
                RT_dim=RT_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bin_size=bin_size,
                max_mz_range=max_mz_range,
                num_layer=9,
                spec_mode='chrom'
            )

        elif self.mode_spec == "chrom_split":
            self.peptide_chrom_embedding = ChromatogramTransformer(
                d_model=d_model,
                device=device,
                n_head=n_head,
                RT_dim=RT_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bin_size=bin_size,
                max_mz_range=max_mz_range,
                num_layer=2,
                spec_mode='chrom'
            )
            self.frag_chrom_embedding = ChromatogramTransformer(
                d_model=d_model,
                device=device,
                n_head=n_head,
                RT_dim=RT_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bin_size=bin_size,
                max_mz_range=max_mz_range,
                num_layer=2,
                spec_mode='chrom'
            )

            self.ts_encoding = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    batch_first=True,
                    dropout=dropout,
                    dim_feedforward=dim_feedforward
                ), num_layers=4)


            self.frag_chrom_proj = nn.Sequential(
                nn.Linear(RT_dim, d_model),
                nn.LayerNorm(d_model)
            )

        elif self.mode_spec == "spec":
            self.spec_embedding = ChromatogramTransformer(
                d_model=d_model,
                device=device,
                n_head=n_head,
                RT_dim=RT_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bin_size=bin_size,
                max_mz_range=max_mz_range,
                num_layer=9,
                spec_mode='spec'
            )

        elif self.mode_spec == "spec_RT":
            self.spec_embedding = ChromatogramTransformer(
                d_model=d_model,
                device=device,
                n_head=n_head,
                RT_dim=RT_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bin_size=bin_size,
                max_mz_range=max_mz_range,
                num_layer=9,
                spec_mode='spec_RT'
            )

        elif self.mode_spec == "spec_split_RT":
            self.peptide_spec_embedding = ChromatogramTransformer(
                d_model=d_model,
                device=device,
                n_head=n_head,
                RT_dim=RT_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bin_size=bin_size,
                max_mz_range=max_mz_range,
                num_layer=2,
                spec_mode='spec_RT'
            )
            self.frag_spec_embedding = ChromatogramTransformer(
                d_model=d_model,
                device=device,
                n_head=n_head,
                RT_dim=RT_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                bin_size=bin_size,
                max_mz_range=max_mz_range,
                num_layer=2,
                spec_mode='spec_RT'
            )
            self.ts_encoding = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_head,
                    batch_first=True,
                    dropout=dropout,
                    dim_feedforward=dim_feedforward
                ), num_layers=4)


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

        peptide_feature = peptide_feature[:, :1, :].squeeze(1)
        spec_feature = torch.max(spec_feature, dim=1, keepdim=False).values

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

        if self.mode_peptide == 'trans':
            peptide_feature = self.peptide_embedding(peptides, modifications)

        elif self.mode_peptide == 'esm2':
            peptide_feature = self.peptide_embedding(peptides)

        return peptide_feature

    def spec_forward(self, data):
        peptide_chrom = data["peptide_chrom"]
        peptide_mz = data["peptide_mz"]
        peptide_RT = data["peptide_RT"]
        frag_chrom = data["fragment_chrom"]
        frag_mz = data["fragment_mz"]
        frag_RT = data["fragment_RT"]
        if self.mode_spec == "chrom":
            chrom = torch.cat([peptide_chrom, frag_chrom], dim=1)  # [B, 14, RT_dim]
            mz = torch.cat([peptide_mz, frag_mz], dim=-1)
            spec_feature = self.chrom_embedding(chrom, mz, peptide_RT)
        elif self.mode_spec == "chrom_split":
            peptide_feature = self.peptide_chrom_embedding(peptide_chrom, peptide_mz, peptide_RT)
            frag_feature= self.frag_chrom_embedding(frag_chrom, frag_mz, frag_RT)
            spec_feature = self.ts_encoding(torch.cat([peptide_feature, frag_feature], dim=1))
        elif self.mode_spec == "spec":
            spec = torch.cat([peptide_chrom, frag_chrom], dim=1)  # [B, 14, RT_dim]
            mz = torch.cat([peptide_mz, frag_mz], dim=-1)
            spec_feature = self.spec_embedding(spec, mz, peptide_RT)
        elif self.mode_spec == "spec_RT":
            spec = torch.cat([peptide_chrom, frag_chrom], dim=1)  # [B, 14, RT_dim]
            mz = torch.cat([peptide_mz, frag_mz], dim=-1)
            spec_feature = self.spec_embedding(spec, mz, peptide_RT)
        elif self.mode_spec == "spec_split_RT":
            peptide_spec_feature = self.peptide_spec_embedding(peptide_chrom, peptide_mz, peptide_RT)
            frag_spec_feature = self.frag_spec_embedding(frag_chrom, frag_mz, frag_RT)
            spec_feature = self.ts_encoding(torch.cat([peptide_spec_feature, frag_spec_feature], dim=1))
        return spec_feature

class ChromatogramTransformer(nn.Module):
    def __init__(
            self,
            d_model = 512,
            RT_dim = 32,
            bin_size=2500,
            max_mz_range=5000,
            n_head = 16,
            dim_feedforward = 1024,
            dropout=0.2,
            num_layer = 8,
            device='cuda',
            spec_mode='spec_RT'
    ):
        super().__init__()
        self.mode = spec_mode
        self.device = device
        self.d_model = d_model
        self.RT_dim = RT_dim
        self.bin_size = bin_size
        self.max_mz_range = max_mz_range
        self.ts_encoding = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, batch_first=True, dropout=dropout, dim_feedforward=dim_feedforward), num_layers=num_layer)
        self.position_emb = PositionEmbedding(device=device)
        self.spec_mlp = nn.Sequential(
            nn.Linear(self.RT_dim, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        self.bin_mlp = nn.Sequential(
            nn.Linear(self.bin_size, self.d_model),
            nn.LayerNorm(self.d_model)
        )


    def mz_emb(self, mz, shape, specify_position=False):
        """
        :param mz: [batch, ion_num]
        :param shape: [batch, ion_num, d_model]
        :return: position_emb: [batch, seq, d_model]
        """
        if specify_position:
            return self.position_emb.embedding(shape=shape, position=mz.unsqueeze(-1), position_specify=True)
        else:
            return self.position_emb.embedding(shape)


    def RT_emb(self, RT, shape, specify_position=False):
        """
        :param RT: [batch, RT_dim]
        :param shape: [batch, RT_dim, d_model]
        :return: position_emb:[batch, RT_dim, d_model]
        """
        if specify_position:
            return self.position_emb.embedding(shape=shape, position=RT.unsqueeze(-1), position_specify=True)
        else:
            return self.position_emb.embedding(shape)


    def spec_bin_emb(self, spec, mz, add_noise=False):
        """
        :param spec: [batch, ion_num, RT_dim]
        :param mz: [batch, ion_num]
        :param max_bin_mz: int/float
        :return: spec_feature: [batch, RT_dim, bin_size]
        """
        batch, ion_num, RT_dim = spec.shape
        device = spec.device
        target = torch.zeros(batch, RT_dim, self.bin_size).to(device)
        noise = torch.rand(batch, RT_dim, self.bin_size).to(device)
        indices = (mz / self.max_mz_range * self.bin_size).long()  # [B, ion_num]
        indices = indices.unsqueeze(1).repeat(1, RT_dim, 1)  # [B, RT_dim, ion_num]
        indices = torch.clamp(indices, 0, self.bin_size - 1) # 确保索引在有效范围内
        spec = spec.permute(0,2,1) #[batch, RT_dim, ion_num]
        target = target.scatter_add(dim = -1, index = indices, src = spec)
        if add_noise:
            target += noise * 0.1
        return target


    def forward(self, spec, mz, RT):
        """
        :param spec: [batch, ion_num, RT_dim]
            cls(RT): [batch, RT_dim]
            mz: [batch, ion_num]
        :return: [batch, ion_num, d_model]
        """
        if self.mode == 'chrom':
            B, ion_num, RT_dim = spec.shape
            spec = torch.cat([RT.unsqueeze(1), spec], dim=1)
            feature = self.spec_mlp(spec)
            feature[:, 1:, :] += self.mz_emb(mz=mz, shape=(B, ion_num, self.d_model), specify_position=True)
            feature = self.ts_encoding(feature)
        elif self.mode == 'spec':
            B, ion_num, RT_dim = spec.shape
            spec = self.spec_bin_emb(spec=spec, mz=mz) #[batch, RT_dim, bin_size]
            feature = self.bin_mlp(spec)
            feature = self.ts_encoding(feature)
        elif self.mode == 'spec_RT':
            B, ion_num, RT_dim = spec.shape
            spec = self.spec_bin_emb(spec=spec, mz=mz)  # [batch, RT_dim, bin_size]
            feature = self.bin_mlp(spec)
            feature += self.RT_emb(RT, (B, RT_dim, self.d_model), specify_position=False)
            feature = self.ts_encoding(feature)
        return feature

class PeptideTransformer(nn.Module):
    def __init__(
            self,
            d_model = 512,
            n_head = 16,
            device = 'cpu',
            dim_feedforward = 2048,
            dropout = 0.25,
            num_layer = 8,
            peptide_vocab = "./peptide_vocab.txt",
            modification_vocab = "./modification_vocab.txt"
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.peptide_tokenizer = BertTokenizer(vocab_file=peptide_vocab)
        self.modification_tokenizer = BertTokenizer(vocab_file=modification_vocab)
        self.word_emb = nn.Embedding(num_embeddings=32, embedding_dim=d_model)
        self.modification_emb = nn.Embedding(num_embeddings=8, embedding_dim=d_model)
        self.position_emb = PositionEmbedding(device=self.device)
        self.ts_encoding = nn.ModuleList([Transformer(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_layer)])

    def modification_embedding(self, modifications, shape):
        """
        :param modifications: (B) string type
        :return: [batch, seq, d_model]
        """
        B, seq_num, d_model = shape
        modification_str = [['None' for _ in range(seq_num)]  for _ in range(B)]
        for index, modification in enumerate(modifications):
            if modification == '':
                continue
            for item in modification.split(','):
                key, value = item.split(':')
                key = int(key)
                new_str = re.sub(r'\[.*?\]', '', value)
                modification_str[index][key] = new_str
        modification_id = [self.modification_tokenizer.encode(' '.join(modification), add_special_tokens=False) for modification in modification_str]
        modification_id = torch.tensor(modification_id).to(self.device) #[B, seq]
        modification_feature = self.modification_emb(modification_id)
        return modification_feature

    def word_embedding(self, peptides):
        """
            :param peptides: (batch) string
            :return: [batch, seq, d_model]
        """
        input_ids = [torch.tensor(self.peptide_tokenizer.encode(peptide, add_special_tokens=True)).to(self.device) for peptide in peptides]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=31).to(self.device) #[B, seq]
        B, seq = input_ids.shape
        mask = (input_ids != 31).to(self.device) #[B, seq]
        mask = mask.unsqueeze(-1) * mask.unsqueeze(1) #[B, seq, seq]
        mask[torch.eye(seq).unsqueeze(0).repeat(B,1,1) == 1] = 1
        mask = mask.float()
        mask[mask == 0] = float('-inf')
        word_feature = self.word_emb(input_ids)
        return word_feature, mask

    def forward(self, peptides, modifications):
        """
        :param peptide: (batch)
               modifications: (batch)
        :return: [batch, seq, d_model]
        """
        word_emb, mask = self.word_embedding(peptides)
        modification_emb = self.modification_embedding(modifications, word_emb.shape)
        position_emb = self.position_emb.embedding(word_emb.shape)
        feature = word_emb + modification_emb + position_emb
        for layer in self.ts_encoding:
            feature, mask = layer(feature, mask)
        return feature

class Transformer(nn.Module):
    def __init__(self, d_model = 512, n_head = 64, dim_feedforward = 2048, dropout = 0.25):
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
        mask = attn_mask.unsqueeze(1).repeat(1,self.n_head, 1, 1).reshape(B*self.n_head, seq, seq)
        attn_output, attn_weight = self.multiheadAttention(query, key, value, attn_mask = mask)
        x = self.norm_layer(x + attn_output)
        x = self.norm_layer(x + self.feedforward_layer(x))

        return x, attn_mask

class PositionEmbedding():
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

    def embedding(self, shape, position=None, position_specify=False):
        """
        :param shape: (batch, seq, d_model)
        :return:
        """
        if position_specify:
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

class ESMEmbdedding(nn.Module):
    def __init__(self, d_model=512, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.mlp = nn.Sequential(
            nn.Linear(1280, self.d_model),
            nn.LayerNorm(self.d_model)
        )


    def forward(self, peptides):
        data = [ (" ", peptide) for peptide in peptides]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        feature = results["representations"][33].to(self.device) #[B, token_len, 1280]
        feature = torch.mean(feature, dim=1, keepdim=True)
        feature = self.mlp(feature)
        return feature



if __name__ == "__main__":
    model = ChromatogramTransformer(d_model=512, n_head=64, bin_size=2500, RT_dim=100, dim_feedforward=1024, dropout=0.3, num_layer=2)

    spec = torch.randn(64, 10, 100)
    mz = torch.rand(64, 10) * 5000
    RT = torch.arange(0, 100, 1).unsqueeze(0).repeat(64, 1)


    model(spec, mz, RT)
    peptide = ["L N P W Y H F L M Q V A P P K", 'N L E V G R L L N I S M T M D S P K']
    modifications = ["11:Oxidation[M]", "16:Oxidation[M]"]



    print(model.position_emb(torch.tensor([[2,5,6],[6,3,7]]), (2, 3, 5)))




