import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
from scipy import io


def create_dataset(data_file_path:str, mode:str):

    # gather dataset information from data.yaml file
    with open(data_file_path, 'r') as stream:
        try:
            data_file = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    dataset_name, img_path, gt_path, name_mat_img, name_mat_gt = data_file['name'], data_file['path_img'], data_file['path_gt'], data_file['mat_img_name'], data_file['mat_gt_name']
    img = io.loadmat(img_path)[name_mat_img]
    gt = io.loadmat(gt_path)[name_mat_gt]

    num_features = int(data_file['num_features'])
    labels = data_file['LABELS']
    color_palette = data_file['COLOR_PALETTE']


    if mode == '2D':
        dataset = Dataset2D(img, gt, labels)
    elif mode == '3D':
        dataset = Dataset3D(img, gt, labels)
    
    return dataset_name, dataset, num_features, color_palette


class Dataset3D(Dataset):

    def __init__(self, data:np.ndarray, gt:np.ndarray, labels_list:dict):
        super().__init__()

        self.data = data
        self.gt = gt
        self.labels_list = labels_list

        # create a list of pixels represented by a tuple of their coordinates
        self.pixels = []
        for cl in list(self.labels_list.keys())[1:]: # from 2nd element to ignore pixels of unknown class 0
            ys, xs = np.where(self.gt == cl) # yields an array of positons matching the current class
            for y, x in zip(ys, xs):
                self.pixels.append(np.array([y, x]))

    def __len__(self):
        return len(self.pixels)

    def _get_px_label(self, pos):
        # a function to get label from coordinates
        label = np.zeros(shape=((len(self.labels_list))))
        label[self.gt[pos[0], pos[1]]] = 1
        return torch.Tensor(label)

    def __getitem__(self, index):
        # build a 3D block of the surrounding pixels
        block = np.zeros(shape=(5,5,self.data.shape[2]))
        pos = self.pixels[index]

        for j in range(pos[0]-2,pos[0]+3):
            for i in range(pos[1]-2,pos[1]+3):
                if i>0 and j>0 and i<self.data.shape[1]-1 and j<self.data.shape[0]-1:
                    block[j-(pos[0]-2),i-(pos[1]-2)] = self.data[j,i]
        label = self._get_px_label(pos)

        return torch.Tensor(block), label


class IndianPinesDataset2D(Dataset):

    def __init__(self, data:np.ndarray, gt:np.ndarray, labels_list:dict):
        super().__init__()

        # create list of pixels represented by a tuple of their coordinates
        self.data = data
        self.gt = gt
        self.labels_list = labels_list
        pixels_by_cl = {}

        self.pixels = []
        for cl in self.labels_list.keys(): # from 2nd element to ignore pixels of unknown class 0
            if cl:
                ys, xs = np.where(self.gt == cl) # yields an array of ys and one of xs
                ##ys = np.where(self.gt == cl)[0][0]
                ##xs = np.where(self.gt == cl)[1][0]
                #pixels_by_cl[str(cl)] = locs
                
                ##self.pixels.append(np.array([ys, xs]))
                for y, x in zip(ys[:30], xs[:30]):
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
                if i>0 and j>0 and i<self.data.shape[1]-1 and j<self.data.shape[0]-1:
                    block[j-(pos[0]-2),i-(pos[1]-2)] = self.data[j,i]
        #block = self.data[pos[0]-2:pos[0]+3, pos[1]-2:pos[1]+3]
        label = self._get_px_label(pos)

        return torch.Tensor(block).transpose(dim0=2, dim1=0), label

class Dataset2D(Dataset):

    def __init__(self, data:np.ndarray, gt:np.ndarray, labels_list:dict):
        super().__init__()

        self.data = data
        self.gt = gt
        self.labels_list = labels_list

        # create a list of pixels represented by a tuple of their coordinates
        self.pixels = []
        for cl in list(self.labels_list.keys())[1:]: # from 2nd element to ignore pixels of unknown class 0
            ys, xs = np.where(self.gt == cl) # yields an array of positons matching the current class
            for y, x in zip(ys, xs):
                self.pixels.append(np.array([y, x]))

    def __len__(self):
        return len(self.pixels)

    def _get_px_label(self, pos):
        # a function to get label from coordinates
        label = np.zeros(shape=((len(self.labels_list))))
        label[self.gt[pos[0], pos[1]]] = 1
        return torch.Tensor(label)

    def __getitem__(self, index):
        # build a 3D block of the surrounding pixels
        block = np.zeros(shape=(5,5,self.data.shape[2]))
        pos = self.pixels[index]

        for j in range(pos[0]-2,pos[0]+3):
            for i in range(pos[1]-2,pos[1]+3):
                if i>0 and j>0 and i<self.data.shape[1]-1 and j<self.data.shape[0]-1:
                    block[j-(pos[0]-2),i-(pos[1]-2)] = self.data[j,i]
        label = self._get_px_label(pos)

        return torch.Tensor(block).transpose(dim0=2, dim1=0), label