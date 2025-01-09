import numpy as np
from torch.utils.data import Dataset
import torch




class IndianPinesDataset(Dataset):

    def __init__(self, data:np.ndarray, gt:np.ndarray, labels_list:dict):
        super().__init__()

        # create list of pixels represented by a tuple of their coordinates
        self.data = data
        self.gt = gt
        self.labels_list = labels_list
        pixels_by_cl = {}

        self.pixels = []
        for cl in self.labels_list.keys(): # from 2nd element to ignore pixels of unknown class
            if cl:
                ys, xs = np.where(self.gt == cl) # yields an array of ys and one of xs
                #pixels_by_cl[str(cl)] = locs
                for y, x in zip(ys, xs):
                    self.pixels.append(np.array([y, x]))

    def __len__(self):
        return len(self.pixels)

        # build a method to get label from coordinates
    def _get_px_label(self, pos):
        label = np.zeros(shape=((len(self.labels_list))))
        label[self.gt[pos[0], pos[1]]] = 1
        return torch.Tensor(label)
        # on __get_item__ method, build a 3D block of the surrounding pixels

    def __getitem__(self, index):
        block = np.zeros(shape=(5,5,200))
        pos = self.pixels[index]

        for j in range(pos[0]-2,pos[0]+3):
            for i in range(pos[1]-2,pos[1]+3):
                if i>0 and j>0 and i<self.data.shape[1]-1 and j<self.data.shape[1]-1:
                    block[j-(pos[0]-2),i-(pos[1]-2)] = self.data[j,i]
        #block = self.data[pos[0]-2:pos[0]+3, pos[1]-2:pos[1]+3]
        label = self._get_px_label(pos)

        return torch.Tensor(block), label