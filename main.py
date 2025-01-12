import os
import numpy as np
from scipy import io
from utils import viz_img, viz_results, create_block, plot_distribution
import yaml
from random import shuffle
from datetime import datetime
import argparse
from datasets import IndianPinesDataset3D, IndianPinesDataset2D, PaviaUniversityDataset2D
from models import PixelClassifier3D, PixelClassifier2D
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from tqdm import tqdm
from torchvision import transforms


def main(args):
    dataset_name = args.dataset
    train = args.train
    nn_mode = args.nn_mode
    verbose = args.verbose

    if dataset_name == "indian_pines":
        path_img = "./data/indian_pines/Indian_pines_corrected.mat"
        path_gt = "./data/indian_pines/Indian_pines_gt.mat"
        path_data = "./data/indian_pines/data.yaml"
        file = io.loadmat(path_img)
        img = file["indian_pines_corrected"]
        file = io.loadmat(path_gt)
        gt = file["indian_pines_gt"]
        num_features = 200
    elif dataset_name == "paviau":
        path_img = "./data/pavia_university/PaviaU.mat"
        path_gt = "./data/pavia_university/PaviaU_gt.mat"
        path_data = "./data/pavia_university/data.yaml"
        file = io.loadmat(path_img)
        img = file["paviaU"]
        file = io.loadmat(path_gt)
        gt = file["paviaU_gt"]
        num_features = 103

    save_path = "./runs/"+dataset_name+"_"+datetime.now().strftime('%m%d%H%M')
    os.makedirs(save_path)
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
    color_palette = file['COLOR_PALETTE']


    print("Creating dataset from image")
    if nn_mode == "3D":
        dataset = IndianPinesDataset3D(img, gt, labels)
    elif nn_mode == "2D":
        if dataset_name == "indian_pines":
            dataset = IndianPinesDataset2D(img, gt, labels)
        elif dataset_name == "paviau":
            dataset = PaviaUniversityDataset2D(img, gt, labels)

    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    plot_distribution(save_path+'/data_distribution.png', labels, train_loader, val_loader)

    if nn_mode == "3D":
        model = PixelClassifier3D(nc=len(dataset.labels_list))
    elif nn_mode == "2D":
        model = PixelClassifier2D(nc=len(dataset.labels_list), num_features=num_features)

    #block, label = next(iter(loader))
    #print(f"Block data : {block.shape}")
    #print(f"Label : {label}")

    #output = model(block)
    #print(output)
    #print(output.shape)

    if train:
        best_score = 0
        lr = 0.001
        num_epochs = 30
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

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
            y_test, y_pred = [], []
            with torch.no_grad():
                for block, label in val_loader:
                    if nn_mode == "3D":
                        block = block.unsqueeze(dim=1)
                    output = model(block)
                    if output.shape[0]!=1:
                        output=output.squeeze()
                    vloss = criterion.forward(output, label)
                    epoch_vloss += vloss.item()
                    for o, l in zip(output, label):
                        y_pred.append(o.argmax())
                        y_test.append(l.argmax())
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            avg_loss = epoch_loss/len(train_loader)
            avg_vloss = epoch_vloss/len(val_loader)

            print(f"F1-score : {f1:.3f}  |  Training / Validation losses : {avg_loss:.3f} / {avg_vloss:.3f}")
            # Save
            if best_score < f1:
                print("Saving model")
                best_score = f1
                torch.save(model.state_dict(), os.path.join(save_path, "best.pt"))

    tp = 0
    fp = 0
    y_test, y_pred = [], []
    with torch.no_grad():
        for block, label in val_loader:
            if nn_mode == "3D":
                block = block.unsqueeze(dim=1)
            output = model(block)
            for o, l in zip(output.squeeze(), label.float()):
                y_pred.append(o.argmax())
                y_test.append(l.argmax())
                if o.argmax()==l.argmax():
                #if float(round(o.item())) == l.item():
                    tp += 1
                else :
                    fp += 1
    print(f"TP : {tp}/{tp+fp}")

    # Confusion Matrix
    #cm = confusion_matrix(y_test, y_pred)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels.keys())
    #disp.plot(cmap=plt.cm.RdPu)
    #plt.title('Confusion Matrix')
    #plt.savefig(save_path+"/confusion_matrix.png")

    # Show predictions only for labeled pixels
    print("Predicting on all labeled pixels...")
    results = np.zeros(shape=gt.shape)
    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            if gt[j, i]:
                pos = np.array([j, i])
                block = create_block(img, pos, num_features)
                if nn_mode == '2D':
                    block = block.transpose(dim0=2, dim1=0)
                pred = model(block.unsqueeze(dim=0))
                results[j, i] = pred.argmax()
    viz_results(save_path+"/results.png", gt, results, labels, color_palette)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        type=str,
                        help="Specify which dataset to work on. Accepts indian_pines or paviau.")
    parser.add_argument("--train",
                        action="store_true",
                        help="Specify if the model should be trained.")
    parser.add_argument("--nn_mode",
                        type=str,
                        help="Specify the dimensions of the model. Accepts 2D or 3D.")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Detailed results in terminal.")    
    args = parser.parse_args()

    main(args)