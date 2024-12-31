import random
import torch.nn as nn
import pickle
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

class MSDataset(nn.Module):
    def __init__(
            self,
            file_path,
            data_checker,
            device="cpu",
            train_ratio=0.8,
            is_train=True,
            divide_charge = True,
            channel_norm = False
    ):
        super().__init__()
        self.device = device
        self.divide_charge = divide_charge
        self.channel_norm = channel_norm
        self.data = []
        if os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                self.data = pickle.load(file)
        elif os.path.isdir(file_path):
            for file in os.listdir(file_path):
                if file.endswith('.pkl') and os.path.isfile(os.path.join(file_path, file)):
                    with open(os.path.join(file_path, file), "rb") as file:
                        self.data += pickle.load(file)
        else:
            print(f"Error: {file_path} 既不是文件也不是文件夹")
        
        self.data = [data for data in self.data if data_checker.check(data)]

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
                "modification": modification,
                "fragment_chrom": frag_chrom.float().to(self.device),
                "fragment_mz": frag_mz.float().to(self.device),
                "fragment_RT": frag_RT.float().to(self.device),
                "fragment_ion_type": frag_type_code.float().to(self.device),
                "label": label.float().to(self.device)}

    def iontype_convert(self, frag_type):
        charge = [s.count('+') + s.count('-') for s in frag_type]
        code = []
        for ion in frag_type:
            if ion.find('b') >= 0:
                code.append(float(''.join(filter(str.isdigit, ion))))
            elif ion.find('y') >= 0:
                code.append(-1 * float(''.join(filter(str.isdigit, ion))))
            else:
                code.append(0.0)
        return torch.tensor(code), torch.tensor(charge)


class DataChecker:
    def __init__(self, RT_dim, ion_num):
        self.RT_dim = RT_dim
        self.ion_num = ion_num

    def check(self, data):
        # 检查 'pre' 和 'frag' 的 'chrom' 长度是否一致，且是否等于 RT_dim
        pre_chrom_len = len(data['pre']['chrom'][0])
        frag_chrom_len = len(data['frag']['chrom'][0])
        pre_RT_len = len(data['pre']['RT'])
        frag_RT_len = len(data['frag']['RT'])

        # 校验 'pre' 和 'frag' 的 'chrom' 列表的长度是否一致且等于 RT_dim
        if pre_chrom_len != frag_chrom_len or pre_chrom_len != self.RT_dim:
            return False
        
        # 校验 'pre' 和 'frag' 的 'RT' 列表的长度是否一致且等于 RT_dim
        if pre_RT_len != frag_RT_len or pre_RT_len != self.RT_dim:
            return False
        
        # 校验 'frag' 的 'chrom' 和 'mz' 长度是否一致，且等于 ion_num
        frag_chrom_mz_len = len(data['frag']['chrom'])
        frag_mz_len = len(data['frag']['mz'])

        if frag_chrom_mz_len != frag_mz_len or frag_chrom_mz_len != self.ion_num:
            return False

        return True


if __name__ == "__main__":
    # dataset1 = MSDataset(".\\DatasetFormer\\data_12_15.pkl", is_train=False)
    # loader1 = DataLoader(dataset=dataset1, batch_size=2, shuffle=True)
    # for index, data in enumerate(loader1):
    #     print(data)
    #     if index > 2:
    #         break
    a = [1,2,3]
    b = [4,5,6]
    print(a + b)