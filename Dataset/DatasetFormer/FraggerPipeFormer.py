import pickle
import random
import argparse
import os

# chormatogram [4, RT_dim]            m/z [4]             RT [RT_dim]                   peptide str         charge int     modification     label
# chormatorgram [ion_num, RT_dim]     m/z [ion_num]       fragment type [ion_num]       RT    [RT_dim]      peptide        charge           modification     label
class PickleData():
    def __init__(self, pre_filename, frag_filename):
        with open(pre_filename, "rb") as pre_file:
            self.preRawData = pickle.load(pre_file)
        with open(frag_filename, "rb") as frg_file:
            self.fragRawData = pickle.load(frg_file)

        ## check validate data
        if (len(self.preRawData) != len(self.fragRawData)):
            print("Error: the data number of precursor is not same with number of fragment !!!")
            return

        self.preData = []
        self.fragData = []
        self.label = []
        self.length = len(self.preRawData)

        for i in range(self.length):

            if not (self.valid_data_check(self.preRawData[i], self.fragRawData[i])):
                self.preData.clear()
                self.fragData.clear()
                self.label.clear()
                self.length = 0
                return

            self.preData.append({"chrom": self.preRawData[i][0],
                                 "mz": self.preRawData[i][1],
                                 "RT": self.preRawData[i][2],
                                 "peptide": self.preRawData[i][3],
                                 "charge": self.preRawData[i][4],
                                 "modification": self.preRawData[i][5]})
            self.fragData.append({"chrom": self.fragRawData[i][0],
                                  "mz": self.fragRawData[i][1],
                                  "IonType": self.fragRawData[i][2],
                                  "RT": self.fragRawData[i][3]})

            self.label.append(self.preRawData[i][6])

    def valid_data_check(self, preData, fragData) -> bool:
        if (preData[3] != fragData[4]):
            print("Error: the peptide of precursor is not same with fragment !!!")
            return False

        if (preData[4] != fragData[5]):
            print("Error: the peptide charge of precursor is not same with fragment !!!")
            return False

        if (preData[5] != fragData[6]):
            print("Error: the modification of precursor is not same with fragment !!!")
            return False

        if (preData[6] != fragData[7]):
            print("Error: the label of precursor is not same with fragment !!!")
            return False

        if(len(preData[0][0]) != len(preData[2]) or len(fragData[0][0]) != len(fragData[3]) or len(preData[2]) != len(fragData[3])):
            print("Error: the RT dim of precursor is not same with fragment !!!")
            return False

        if (len(preData[0]) != len(preData[1]) or len(fragData[0]) != len(fragData[1]) or len(fragData[0]) != len(fragData[2])):
            print("Error: the ion number of precursor or fragment is not same !!!")
            return False

        return True

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {'pre': self.preData[index], 'frag': self.fragData[index], 'label': self.label[index]}


class DataFormer:
    def __init__(self, PreFilename, FragFilename, filter_label = 1):
        self.data = []
        if PreFilename is not None and FragFilename is not None:
            self.data = [item for item in PickleData(PreFilename, FragFilename) if item['label'] == filter_label]

        #需要处理负样例的label为0
        if filter_label == -1:
            for i in range(len(self.data)):
                self.data[i]['label'] = 0

        print(f"Data Length: {len(self.data)}, Data Label: {filter_label}")

    def dump(self, out_filename, shuffle = True):
        if shuffle:
            random.shuffle(self.data)

        with open(out_filename, 'wb') as file:
            pickle.dump(self.data, file)

        print(f"{len(self.data)} data output to {out_filename}")


class DataConsolation:
    def __init__(self, pos_path, neg_path):
        # 读取正样本和负样本路径下的所有pkl文件
        self.pos_data = self._load_data_from_path(pos_path)
        self.neg_data = self._load_data_from_path(neg_path)

    def _load_data_from_path(self, path):
        """ 从指定路径加载所有pkl文件 """
        data = []
        for file_name in os.listdir(path):
            if file_name.endswith('.pkl'):
                file_path = os.path.join(path, file_name)
                with open(file_path, 'rb') as f:
                    data += pickle.load(f)
        return data

    def check(self):
        """ 检查正样本label是否为1，负样本label是否为0 """
        pos_check = all(sample['label'] == 1 for sample in self.pos_data)
        neg_check = all(sample['label'] == 0 for sample in self.neg_data)

        if not pos_check:
            print("正样本标签不全为1")
        if not neg_check:
            print("负样本标签不全为-1")

        return pos_check, neg_check

    def dump(self):
        """ 合并正负样本，shuffle并控制正负样本数量相同，多余的丢掉 """
        # 调整正负样本数目一致
        min_len = min(len(self.pos_data), len(self.neg_data))
        pos_data_sampled = random.sample(self.pos_data, min_len)
        neg_data_sampled = random.sample(self.neg_data, min_len)
        # 合并数据并shuffle
        combined_data = pos_data_sampled + neg_data_sampled
        random.shuffle(combined_data)

        return combined_data


if __name__ == '__main__':
    x = [1,2,3]
    print(random.shuffle(x))