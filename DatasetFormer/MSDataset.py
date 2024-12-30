import random
import torch.nn as nn
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader

class MSDataset(nn.Module):
    def __init__(
            self,
            filename,
            device="cpu",
            train_ratio=0.8,
            is_train=True,
            label_split=0,
            divide_charge = 1,
            channel_norm = 0
    ):
        super().__init__()
        self.device = device
        self.divide_charge = divide_charge
        self.channel_norm = channel_norm

        with open(filename, "rb") as file:
            self.data = pickle.load(file)

        if label_split == 1:
            self.data = list(filter(lambda x: x['label'] == 1, self.data))
        elif label_split == -1:
            self.data = list(filter(lambda x: x['label'] == 0, self.data))

        if is_train:
            number = round(len(self.data) * train_ratio)
            self.data = self.data[:number]
        else:
            number = round(len(self.data) * train_ratio)
            self.data = self.data[number:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pre = self.data[index]['pre']
        frag = self.data[index]['frag']

        # label
        label = torch.tensor(self.data[index]['label'])

        #chrom process
        peptide_chrom = torch.tensor(pre['chrom'])
        frag_chrom = torch.tensor(frag['chrom'])
        if self.channel_norm:
            peptide_chrom = peptide_chrom / ((torch.max(peptide_chrom, dim=-1).values).unsqueeze(-1) + 1e-4)  # [peptide_num, RT_dim]
            frag_chrom = frag_chrom / ((torch.max(frag_chrom, dim=-1).values).unsqueeze(-1) + 1e-4)  # [ion_num, RT_dim]
        else:
            peptide_chrom = peptide_chrom / ((torch.max(peptide_chrom.reshape(-1), dim=-1).values).unsqueeze(-1) + 1e-4)  # [peptide_num, RT_dim]
            frag_chrom = frag_chrom / ((torch.max(frag_chrom.reshape(-1), dim=-1).values).unsqueeze( -1) + 1e-4)  # [ion_num, RT_dim]

        #mz process
        if self.divide_charge:
            peptide_mz = torch.tensor(pre['mz'])
            frag_mz = torch.tensor(frag['mz'])
            frag_type_code, frag_type_charge = self.iontype_convert(frag['IonType'])
        else:
            peptide_mz = torch.tensor(pre['mz']) * pre['charge']
            frag_mz = torch.tensor(frag['mz'])
            frag_type_code, frag_type_charge = self.iontype_convert(frag['IonType'])
            frag_mz = frag_mz * frag_type_charge

        #RT process
        peptide_RT = torch.tensor(pre['RT'])
        frag_RT = torch.tensor(frag['RT'])

        #peptide && modification
        peptide = pre['peptide']
        peptide = ' '.join(peptide)
        modification = ','.join(f"{key}:{value}" for key, value in pre['modification'].items())


        return {"peptide_chrom": peptide_chrom.float().to(self.device),
                "peptide_mz": peptide_mz.float().to(self.device),
                "peptide_RT": peptide_RT.float().to(self.device),
                "peptide": peptide,
                "fragment_chrom": frag_chrom.float().to(self.device),
                "fragment_mz": frag_mz.float().to(self.device),
                "fragment_RT": frag_RT.float().to(self.device),
                "fragment_ion_type": frag_type_code.float().to(self.device),
                "modification": modification,
                "label": label.float().to(self.device)}

    def iontype_convert(self, frag_type):
        charge = [s.count('+') + s.count('-') for s in frag_type]
        code = [float(''.join(filter(str.isdigit, s))) if s.find('b') >= 0 else -1.0 * float(''.join(filter(str.isdigit, s))) for s in frag_type]
        return torch.tensor(code), torch.tensor(charge)


if __name__ == "__main__":
    dataset1 = MSDataset(".\\DatasetFormer\\data_12_15.pkl", is_train=False)
    loader1 = DataLoader(dataset=dataset1, batch_size=2, shuffle=True)
    for index, data in enumerate(loader1):
        print(data)
        if index > 2:
            break
