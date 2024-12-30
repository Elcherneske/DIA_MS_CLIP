import pickle
import numpy as np

class DataChecker():
    def __init__(self, dataFilename):
        super().__init__()
        with open(dataFilename, "rb") as file:
            self.data = pickle.load(file)

    def RT_dim_range(self):
        RT_dim = [len(data['pre']['chrom'][0]) for data in self.data]

        return {'max': np.max(RT_dim), 'min': np.min(RT_dim)}

    def Ion_number_range(self):
        ion_num = [len(data['frag']['chrom']) for data in self.data]

        return {'max': np.max(ion_num), 'min': np.min(ion_num)}

    def peptide_length_range(self):
        peptide_length = [len(data['pre']['peptide']) for data in self.data]

        return {'max': np.max(peptide_length), 'min': np.min(peptide_length)}



if __name__ == '__main__':
    checker = DataChecker("D:\code\Python\MS\DIA\DatasetFormer\post_data\PDX005573_Fig1_MP-DIA-120min-120kMS1-40W15k_MHRM_R03.pkl")
    print(checker.RT_dim_range())
    print(checker.Ion_number_range())
    print(checker.peptide_length_range())




