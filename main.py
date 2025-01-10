import numpy as np
from scipy import io
from utils import viz_img, viz_results, create_block, plot_distribution
import yaml
from random import shuffle
import argparse
from datasets import IndianPinesDataset3D, IndianPinesDataset2D
from models import PixelClassifier3D, PixelClassifier2D
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms


def main(args):
    train = args.train
    nn_mode = args.nn_mode

    path_img = "./data/indian_pines/Indian_pines_corrected.mat"
    path_gt = "./data/indian_pines/Indian_pines_gt.mat"
    path_data = "./data/indian_pines/data.yaml"
    file = io.loadmat(path_img)
    img = file["indian_pines_corrected"]
    #transform = transforms.Compose([
    #    transforms.Normalize(mean=[0.5]*200, std=[0.225]*200)
    #])
    #img = transform(torch.Tensor(img))
    #print(type(img))
    #print(type(np.float64(np.max(img))))
    #img /= np.float64(np.max(img))
    #img = np.array(img)
    #print(type(img[0,0,0]))
    #img /= np.uint16(9604)

    with open(path_data, 'r') as stream:
        try:
            file = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)
    labels = file['LABELS']

    file = io.loadmat(path_gt)
    gt = file["indian_pines_gt"]

    print("Creating dataset")
    if nn_mode == "3D":
        dataset = IndianPinesDataset3D(img, gt, labels)
    elif nn_mode == "2D":
        dataset = IndianPinesDataset2D(img, gt, labels)

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    plot_distribution('./data_distribution.png', train_loader, val_loader)

    if nn_mode == "3D":
        model = PixelClassifier3D(nc=len(dataset.labels_list))
    elif nn_mode == "2D":
        model = PixelClassifier2D(nc=len(dataset.labels_list))

    #block, label = next(iter(loader))
    #print(f"Block data : {block.shape}")
    #print(f"Label : {label}")

    #output = model(block)
    #print(output)
    #print(output.shape)

    if train:
        lr = 0.001
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        num_epochs = 500
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for block, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()

                if nn_mode == "3D":
                    block = block.unsqueeze(dim=1)
                output = model(block)

                if output.shape[0]!=1:
                    output=output.squeeze()
                loss = criterion.forward(output, label)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            model.eval()
            epoch_vloss = 0.0
            with torch.no_grad():
                for block, label in val_loader:
                    if nn_mode == "3D":
                        block = block.unsqueeze(dim=1)
                    output = model(block)
                    if output.shape[0]!=1:
                        output=output.squeeze()
                    vloss = criterion.forward(output, label)
                    epoch_vloss += vloss.item()
            
            avg_loss = epoch_loss/len(train_loader)
            avg_vloss = epoch_vloss/len(val_loader)

            print(f"Training / Validation losses : {avg_loss} / {avg_vloss}")

    tp = 0
    fp = 0
    with torch.no_grad():
        for block, label in tqdm(val_loader, desc="Predicting all samples"):
            if nn_mode == "3D":
                block = block.unsqueeze(dim=1)
            output = model(block)
            for o, l in zip(output.squeeze(), label.float()):
                if o.argmax()==l.argmax():
                #if float(round(o.item())) == l.item():
                    tp += 1
                else :
                    fp += 1
    print(f"TP : {tp}/{tp+fp}")

    # Show predictions only for labeled pixels
    results = np.zeros(shape=gt.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if gt[i, j]:
                pos = np.array([i, j])
                block = create_block(img, pos)
                if nn_mode == '2D':
                    block = block.transpose(dim0=2, dim1=0)
                pred = model(block.unsqueeze(dim=0))
                results[i, j] = pred.argmax()
    viz_results("./results.png", gt, results, labels)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",
                      action="store_true",
                      help="Specify if the model should be trained.")
    parser.add_argument("--nn_mode",
                      type=str,
                      help="Specify the dimensions of the model. Accepts 2D or 3D.")
    args = parser.parse_args()

    main(args)