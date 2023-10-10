import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import wyxPreData


class WOwnDataset(Dataset):
    def __init__(self,protein):
        super(WOwnDataset, self).__init__()
        Kmer, knf, Y = wyxPreData.all_data(protein)  # 这里取pair太慢，还是要用npy文件读
        pair = np.load('./Datasets/circRNA-RBP/'+protein+'/pair.npy', allow_pickle=True)
        self.Kmer = Kmer
        self.knf = knf
        self.pair = pair
        self.Y = Y
        dict_data = {'Kmer': Kmer, 'knf': knf, 'pair': pair, 'Y': Y}
        self.dict_data = dict_data


    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.dict_data.items()}
        return item

    def __len__(self):
        return len(self.Y)





