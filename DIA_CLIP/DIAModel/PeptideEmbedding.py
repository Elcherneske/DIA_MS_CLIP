import torch
import torch.nn as nn
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import esm
from .CommonModel import PositionEmbedding, MaskTransformer

class PeptideEmbedding(nn.Module):
    def __init__(
            self,
            d_model=512,
            n_head=16,
            device='cpu',
            dim_feedforward=2048,
            dropout=0.25,
            num_layer=8,
            peptide_vocab="./peptide_vocab.txt",
            modification_vocab="./modification_vocab.txt",
            model = 'trans'
    ):
        super().__init__()
        self.model = model
        if self.model == "trans":
            self.peptide_embedding = PeptideTransformerEmbedding(
                d_model=d_model,
                device=device,
                n_head=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                peptide_vocab=peptide_vocab,
                modification_vocab=modification_vocab,
                num_layer=num_layer
            )
        elif self.model == "esm2":
            self.peptide_embedding = ESMEmbedding(d_model=d_model, device=device)

    def forward(self, peptides, modifications):
        """
        :param peptides: (batch) string
        :param modifications: (batch) string
        :return: [batch, 1, d_model]
        """
        if self.model == 'trans':
            peptide_feature = self.peptide_embedding(peptides, modifications)
        elif self.model == 'esm2':
            peptide_feature = self.peptide_embedding(peptides)
        peptide_feature = peptide_feature[:, :1, :]
        return peptide_feature

class PeptideTransformerEmbedding(nn.Module):
    def __init__(
            self,
            d_model=512,
            n_head=16,
            device='cpu',
            dim_feedforward=2048,
            dropout=0.25,
            num_layer=8,
            peptide_vocab="./peptide_vocab.txt",
            modification_vocab="./modification_vocab.txt"
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.peptide_tokenizer = BertTokenizer(vocab_file=peptide_vocab)
        self.modification_tokenizer = BertTokenizer(vocab_file=modification_vocab)
        self.word_emb = nn.Embedding(num_embeddings=32, embedding_dim=d_model)
        self.modification_emb = nn.Embedding(num_embeddings=8, embedding_dim=d_model)
        self.position_emb = PositionEmbedding(device=self.device)
        self.ts_encoding = nn.ModuleList([MaskTransformer(d_model=d_model, n_head=n_head, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_layer)])

    def modification_embedding(self, modifications, shape):
        """
        :param modifications: (B) string type
        :return: [batch, seq, d_model]
        """
        B, seq_num, d_model = shape
        modification_str = [['None' for _ in range(seq_num)] for _ in range(B)]
        for index, modification in enumerate(modifications):
            if modification == '':
                continue
            for item in modification.split(','):
                key, value = item.split(':')
                key = int(key)
                new_str = re.sub(r'\[.*?\]', '', value)
                modification_str[index][key] = new_str
        modification_id = [self.modification_tokenizer.encode(' '.join(modification), add_special_tokens=False) for modification in modification_str]
        modification_id = torch.tensor(modification_id).to(self.device)  # [B, seq]
        modification_feature = self.modification_emb(modification_id)
        return modification_feature

    def word_embedding(self, peptides):
        """
        :param peptides: (batch) string
        :return: [batch, seq, d_model]
        """
        input_ids = [torch.tensor(self.peptide_tokenizer.encode(peptide, add_special_tokens=True)).to(self.device) for peptide in peptides]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=31).to(self.device)  # [B, seq]
        B, seq = input_ids.shape
        mask = (input_ids != 31).to(self.device)  # [B, seq]
        mask = mask.unsqueeze(-1) * mask.unsqueeze(1)  # [B, seq, seq]
        mask[torch.eye(seq).unsqueeze(0).repeat(B, 1, 1) == 1] = 1
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

class ESMEmbedding(nn.Module):
    def __init__(self, d_model=512, device='cuda'):
        super().__init__()
        self.d_model = d_model
        self.device = device
        torch.hub.set_dir('../esm_model/')
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.mlp = nn.Sequential(
            nn.Linear(1280, self.d_model),
            nn.LayerNorm(self.d_model)
        )

    def forward(self, peptides):
        data = [(" ", peptide) for peptide in peptides]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
        feature = results["representations"][33].to(self.device)  # [B, token_len, 1280]
        feature = torch.mean(feature, dim=1, keepdim=True)
        feature = self.mlp(feature)
        return feature

if __name__ == "__main__":
    print(1)