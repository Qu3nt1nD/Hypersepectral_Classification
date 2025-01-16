import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import viz_results, plot_distribution
from models import create_model
from datasets import create_dataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from datetime import datetime
from tqdm import tqdm


def main(args):
    data = args.data
    train = args.train
    resume = args.resume
    batch_size = args.batch
    num_epochs = args.epochs
    lr = args.lr
    nn_mode = args.nn_mode
    full_map = args.full_map
    verbose = args.verbose


    name, dataset, num_features, color_palette = create_dataset(data, nn_mode, skip_classes=[0])
    name = args.name if args.name is not None else name

    save_path = "./runs/"+name+"_"+datetime.now().strftime('%m%d%H%M')
    os.makedirs(save_path)


    print("Creating dataset from image")
    train_dataset, val_dataset = random_split(dataset, [0.9, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    plot_distribution(save_path+'/data_distribution.png', dataset.labels_list, train_loader, val_loader)


    model = create_model(nn_mode, dataset, num_features)

    if resume is not None:
        model.load_state_dict(torch.load(resume, weights_only=True), strict=True)


    if train:
        best_score = 0
        writer = SummaryWriter(log_dir=save_path)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for block, label, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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
                for block, label, _ in val_loader:
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

            writer.add_scalar("Loss/train", avg_loss, epoch)
            writer.add_scalar("Loss/val", avg_vloss, epoch)
            writer.add_scalar("LR/lr", optimizer.param_groups[0]["lr"], epoch)

            print(f"F1-score : {f1:.3f}  |  Training / Validation losses : {avg_loss:.3f} / {avg_vloss:.3f}")
            # Save
            if best_score < f1:
                print("Saving model")
                best_score = f1
                torch.save(model.state_dict(), os.path.join(save_path, "best.pt"))
    
        writer.flush()
        writer.close()

    tp = 0
    fp = 0
    y_test, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for block, label, _ in val_loader:
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
    cm = confusion_matrix(y_test, y_pred, labels=list(dataset.labels_list.keys()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(dataset.labels_list.keys()))
    disp.plot(cmap=plt.cm.RdPu)
    plt.title('Confusion Matrix')
    plt.savefig(save_path+"/confusion_matrix.png")

    # Show predictions only for labeled pixels
    sc = [] if full_map else [0]
    full_dataset = create_dataset(data, nn_mode, skip_classes=sc)[1]
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    shape = full_dataset.gt.shape
    results = np.zeros(shape=shape)
    for block, label, pos in tqdm(full_loader, desc='Predicting on all pixels'):
        if nn_mode == "3D":
            block = block.unsqueeze(dim=1)
        output = model(block)
        if output.shape[0]!=1:
            output=output.squeeze()
        for o, p in zip(output, pos):
            results[p[0], p[1]] = o.argmax()

    viz_results(save_path+"/results.png", dataset.gt, results, dataset.labels_list, color_palette)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        type=str,
                        required=True,
                        help="The path to the data.yaml file.")
    parser.add_argument("--train",
                        action="store_true",
                        help="Specify if the model should be trained.")
    parser.add_argument("--resume",
                        type=str,
                        default=None,
                        help="Path to load an existing model.")
    parser.add_argument("--batch",
                        type=int,
                        default=32,
                        help="Batch size for training. Default 32.")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Epochs to train the model. Default 10.")
    parser.add_argument("--lr",
                        type=int,
                        default=0.001,
                        help="Base learning rate for training. Default O.OO1.")
    parser.add_argument("--nn_mode",
                        type=str,
                        required=True,
                        help="Specify the dimensions of the model. Accepts 2D or 3D.")
    parser.add_argument("--name",
                        type=str,
                        default=None,
                        help="Name of the run log. Default dataset_name_date.")
    parser.add_argument("--full_map",
                        action="store_true",
                        help="Unlabeled pixels will also be predicteed for the example.")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Detailed results in terminal.")    
    args = parser.parse_args()

    main(args)