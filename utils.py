import numpy as np
import matplotlib.pyplot as plt
import torch


def viz_img(img:np.ndarray, gt:np.ndarray, labels_list:dict, rgb_bands:tuple, color_palette:dict):

    img_rgb = np.asarray(img, dtype='float32')
    img_rgb = (img_rgb - np.min(img_rgb)) / (np.max(img_rgb) - np.min(img_rgb))
    img_rgb = img_rgb[:,:,rgb_bands]

    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.title('Image')
    plt.axis('off')


    img_labeled = np.zeros(shape=img_rgb.shape)
    for ind in labels_list.keys():
        cl = labels_list[ind]
        img_labeled[np.where(gt == ind)] = color_palette[cl]
    img_labeled /= 255

    plt.subplot(1,2,2)
    plt.imshow(img_labeled)
    plt.title('Ground-Truth')
    plt.axis('off')

def viz_results(save_path, gt, pred, labels_list, color_palette:dict):
    plt.figure(figsize=(8,4))

    img_gt = np.zeros(shape=(gt.shape[0], gt.shape[1], 3))
    for ind in labels_list.keys():
        cl = labels_list[ind]
        img_gt[np.where(gt == ind)] = color_palette[cl]
    img_gt /= 255

    img_pred = np.zeros(shape=(pred.shape[0], pred.shape[1], 3))
    for ind in labels_list.keys():
        cl = labels_list[ind]
        img_pred[np.where(pred == ind)] = color_palette[cl]
    img_pred /= 255

    plt.subplot(1,2,1)
    plt.imshow(img_gt)
    plt.title('Ground-Truth')
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img_pred)
    plt.title('Prediction')
    plt.axis('off')

    plt.savefig(save_path)

def plot_distribution(save_path, labels_list, dataloader1, dataloader2=None):

    distrib1 = compute_distribution(dataloader1, labels_list)

    # Plot parameters
    x = np.arange(len(distrib1))
    shift = 0 if dataloader2 is None else 0.2
    width = 0.4

    # Plot distribution
    fig, ax = plt.subplots(figsize=(8,8))
    ax.bar(x-shift, distrib1.values(), width, color='orange')

    # Add second plot if needed
    if dataloader2 is not None:
        ax2 = ax.twinx()
        distrib2 = compute_distribution(dataloader2, labels_list)
        ax2.bar(x+shift, distrib2.values(), width, color='green')
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(distrib1.keys())
    ax.tick_params('x', rotation=90)
    fig.tight_layout()
    plt.savefig(save_path)

def compute_distribution(dataloader, labels_list:dict):
    distrib = {}
    for c in labels_list.values():
        distrib[c] = 0
    for _, labels in dataloader:
        for l in labels:
            ind_l = int(l.argmax().item())
            distrib[list(labels_list.values())[ind_l]] += 1
    return distrib
    



def create_block(img, pos, num_features):
    """
    Arguments:
        - pos : (j, i) tuple.
    """
    block = np.zeros(shape=(5,5,num_features))

    for j in range(pos[0]-2,pos[0]+3):
        for i in range(pos[1]-2,pos[1]+3):
            if i>0 and j>0 and i<img.shape[1]-1 and j<img.shape[0]-1:
                block[j-(pos[0]-2),i-(pos[1]-2)] = img[j,i]

    block = torch.Tensor(block)

    return block

