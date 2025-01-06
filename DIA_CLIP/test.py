import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from MSDataset import MSDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def pearson_correlation(x, y):
    # 计算均值
    mean_x = torch.mean(x, dim=-1)
    mean_y = torch.mean(y, dim=-1)

    # 计算分子和分母
    numerator = torch.sum((x - mean_x.unsqueeze(-1)) * (y - mean_y.unsqueeze(-1)), dim=-1)
    denominator = torch.sqrt(
        torch.sum((x - mean_x.unsqueeze(-1)) ** 2, dim=-1) * torch.sum((y - mean_y.unsqueeze(-1)) ** 2, dim=-1))

    return numerator / (denominator + 1e-6)


def main():
    pre_filename = '../Dataset/xlc_pre_MSFragger.pkl'
    frag_filename = '../Dataset/xlc_frg_MSFragger.pkl'
    RT_dim = 20
    batch_size = 5
    device = 'cpu'
    dataset = MSDataset(pre_filename=pre_filename, frag_filename=frag_filename, RT_num=RT_dim, is_train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    eva_score = torch.tensor([]).to(device)
    label_score = torch.tensor([]).to(device)


def function(x,y):
    x = x+x
    y = y+y
    return x,y




if __name__ == "__main__":
    import torch
    import esm

    # Load ESM-2 model
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3", "K A <mask> I S Q"),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))

    # Look at the unsupervised self-attention map contact predictions
    import matplotlib.pyplot as plt

    for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
        plt.matshow(attention_contacts[: tokens_len, : tokens_len])
        plt.title(seq)
        plt.show()