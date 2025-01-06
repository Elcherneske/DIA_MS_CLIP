import pickle
import random

import torch
import numpy as np
from scipy.signal import savgol_filter


class Preprocessor():
    def __init__(self, dataFilename):
        super().__init__()
        with open(dataFilename, "rb") as file:
            self.data = pickle.load(file)

    def pearson_correlation(self, x, y):
        # 计算均值
        mean_x = torch.mean(x, dim=-1)
        mean_y = torch.mean(y, dim=-1)

        # 计算分子和分母
        numerator = torch.sum((x - mean_x.unsqueeze(-1)) * (y - mean_y.unsqueeze(-1)), dim=-1)
        denominator = torch.sqrt(
            torch.sum((x - mean_x.unsqueeze(-1)) ** 2, dim=-1) * torch.sum((y - mean_y.unsqueeze(-1)) ** 2, dim=-1))

        return numerator / (denominator + 1e-6)

    def dump_data(self, outFilename):
        with open(outFilename, 'wb') as file:
            pickle.dump(self.data, file)

    def interpolation_single(self, data, new_RT_dim):
        pre = data['pre']
        frag = data['frag']

        pre_RT = pre['RT']
        frag_RT = frag['RT']
        pre_chrom = pre['chrom']
        frag_chrom = frag['chrom']

        if len(pre_RT) == new_RT_dim and len(frag_RT) == new_RT_dim:
            return data

        left_RT = np.maximum(pre_RT[0], frag_RT[0])
        right_RT = np.minimum(pre_RT[-1], frag_RT[-1])

        new_RT = np.linspace(left_RT, right_RT, num = new_RT_dim)

        f = np.interp

        pre_chrom = np.array([f(new_RT, pre_RT, chrom) for chrom in pre_chrom])
        frag_chrom = np.array([f(new_RT, frag_RT, chrom) for chrom in frag_chrom])

        pre_chrom[pre_chrom < 0] = 0.0
        frag_chrom[frag_chrom < 0] = 0.0

        data['pre']['chrom'] = pre_chrom
        data['frag']['chrom'] = frag_chrom
        data['pre']['RT'] = pre_RT
        data['frag']['RT'] = frag_RT

        return data

    def interpolation(self, new_RT_dim):
        for i in range(len(self.data)):
            if (i % 10000 == 0):
                print(f"index: {i}, interpolation finish")
            self.data[i] = self.interpolation_single(self.data[i], new_RT_dim=new_RT_dim)

    def slice_middle_single(self, array, slice_num):
        if isinstance(array, list):
            RT_dim = len(array)
            if RT_dim <= slice_num:
                return array
            start = (RT_dim - slice_num) // 2
            end = start + slice_num

            return array[start: end]
        elif isinstance(array, np.ndarray):
            _, RT_dim = array.shape
            if RT_dim <= slice_num:
                return array
            start = (RT_dim - slice_num) // 2
            end = start + slice_num

            return array[:, start: end]

    def slice_middle(self, slice_num = 50):
        for index in range(len(self.data)):
            self.data[index]['pre']['chrom'] = self.slice_middle_single(self.data[index]['pre']['chrom'], slice_num=slice_num)
            self.data[index]['pre']['RT'] = self.slice_middle_single(self.data[index]['pre']['RT'], slice_num=slice_num)
            self.data[index]['frag']['chrom'] = self.slice_middle_single(self.data[index]['frag']['chrom'], slice_num=slice_num)
            self.data[index]['frag']['RT'] = self.slice_middle_single(self.data[index]['frag']['RT'], slice_num=slice_num)

            if index % 10000 == 0:
                print(f"index: {index}, slice finish")

    def filter(self, filter_option):
        filter_data = []
        if filter_option['mode'] == 'ion_num':
            number = filter_option['value']
            for data in self.data:
                if len(data['frag']['chrom']) == number:
                    filter_data.append(data)
            self.data = filter_data

    def preprocess(self, RT_dim, outputFilename, filter_options = None):
        for filter_option in filter_options:
            self.filter(filter_option)
        self.slice_middle(slice_num=RT_dim)
        self.interpolation(new_RT_dim=RT_dim)
        self.dump_data(outputFilename)


if __name__ == "__main__":
    print(1)


