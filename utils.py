import numpy as np
import matplotlib.pyplot as plt
import torch

COLOR_PALETTE = {
    "unknown" : (0,0,0),
    "alfalfa" : (0,71,189),
    "corn-notill" : (2,136,217),
    "corn-mintill" : (7,185,252),
    "corn" : (0,149,67),
    "grass-pastures" : (0,171,58),
    "grass-trees" : (154,240,0),
    "grass-pasture-mowed" : (255,179,0),
    "hay-windrowed" : (255,206,0),
    "oats" : (255,230,59),
    "soybean-notill" : (234,0,52),
    "soybean-mintill" : (253,71,3),
    "soybean-clean" : (255,130,42),
    "wheat" : (130,0,172),
    "woods" : (182,16,191),
    "buildings-grass-trees-drives" : (204,114,245),
    "stone-steel-towers" : (127, 127, 127),
}

def viz_img(img:np.ndarray, gt:np.ndarray, labels_list:dict, rgb_bands:tuple):

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
        img_labeled[np.where(gt == ind)] = COLOR_PALETTE[cl]
    img_labeled /= 255

    plt.subplot(1,2,2)
    plt.imshow(img_labeled)
    plt.title('Ground-Truth')
    plt.axis('off')

def viz_results(save_path, gt, pred, labels_list):
    plt.figure(figsize=(8,4))

    img_gt = np.zeros(shape=(gt.shape[0], gt.shape[1], 3))
    for ind in labels_list.keys():
        cl = labels_list[ind]
        img_gt[np.where(gt == ind)] = COLOR_PALETTE[cl]
    img_gt /= 255

    img_pred = np.zeros(shape=(pred.shape[0], pred.shape[1], 3))
    for ind in labels_list.keys():
        cl = labels_list[ind]
        img_pred[np.where(pred == ind)] = COLOR_PALETTE[cl]
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

def plot_distribution(save_path, dataloader1, dataloader2=None):

    distrib1 = compute_distribution(dataloader1)

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
        distrib2 = compute_distribution(dataloader2)
        ax2.bar(x+shift, distrib2.values(), width, color='green')
    
    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(distrib1.keys())
    ax.tick_params('x', rotation=90)
    fig.tight_layout()
    plt.savefig(save_path)

def compute_distribution(dataloader):
    distrib = {}
    for c in COLOR_PALETTE.keys():
        distrib[c] = 0
    for _, labels in dataloader:
        for l in labels:
            ind_l = int(l.argmax().item())
            distrib[list(COLOR_PALETTE.keys())[ind_l]] += 1
    return distrib
    



def create_block(img, pos):
    block = np.zeros(shape=(5,5,200))

    for j in range(pos[0]-2,pos[0]+3):
        for i in range(pos[1]-2,pos[1]+3):
            if i>0 and j>0 and i<img.shape[0]-1 and j<img.shape[1]-1:
                block[j-(pos[0]-2),i-(pos[1]-2)] = img[j,i]

    block = torch.Tensor(block)

    return block

