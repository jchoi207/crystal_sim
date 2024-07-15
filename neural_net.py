import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import seaborn as sns

from joblib import dump, load
import os


import preprocessing as pp



def dataset_split(features, labels_r, train_size=0.7, val_size=0.15):

    features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_r_tensor = torch.tensor(labels_r, dtype=torch.float32)
    # labels_c_tensor = torch.tensor(labels_c, dtype=torch.float32)
    

    # Combine features and labels into a custom Dataset

    dataset = TensorDataset(features_tensor, labels_r_tensor)
    

    length = len(dataset)
    train_len = int(length * train_size)
    val_len = int(length * val_size)
    test_len = length - train_len - val_len

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len])

    print(train_data)
    return train_data, val_data, test_data


def get_model_name(name, batch_size, total_size, learning_rate, epoch):
    path = "model_{0}_sz{1}_bs{2}_lr{3}_epoch{4}".format(name,
                                                   total_size,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path


def plot_training_curve(train_loss, val_loss, train_accuracy, val_accuracy):
    plt.title("Train vs Validation Loss")
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

    if train_accuracy != None: 
        plt.title("Train vs Validation Accuracy")
        plt.plot(train_accuracy, label="Train")
        plt.plot(val_accuracy, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc='best')
        plt.show()
        

def train_regression(model, train_data, val_data, batch_size=64, learning_rate=0.01, num_epochs=30, checkpoints=True, plot=True, save_model=False):

    torch.manual_seed(1)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size, shuffle=True)
    total_size = int(len(train_loader) + 2 * len(val_loader))
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    start = time.time()

    for epoch in range(num_epochs):
        train_loss = 0.0

        for inputs, labels in train_loader:
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        with torch.no_grad():
            model.eval()
            val_loss = 0.0

            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
            val_loss /= len(val_loader)

            val_losses.append(val_loss)

        # scheduler.step(val_loss)

        if checkpoints:
            print(f"_______________________Epoch {epoch + 1}_______________________")

            print("Train loss: {} | Validation loss: {} ".format(
                train_losses[epoch],
                val_losses[epoch]
            ))
            # print("Train acc: {} | Validation acc: {} ".format(
            #     train_accuracies[epoch],
            #     val_accuracies[epoch]
            # ))

    if not checkpoints:
        print(f"_______________________Epoch {num_epochs}_______________________")

        print("Train loss: {} | Validation loss: {} ".format(
            train_losses[num_epochs-1],
            val_losses[num_epochs-1]
        ))
        # print("Train acc: {} | Validation acc: {} ".format(
        #     train_accuracies[num_epochs-1],
        #     val_accuracies[num_epochs-1]
        # ))

    print('Finished Training')
    elapsed_time = time.time() - start
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    if save_model:
        torch.save(model.state_dict(), 
                   os.path.join('/Users/jonathanchoi/Desktop/GitHub Projects/crystal_sim/', get_model_name(model.name, total_size, batch_size, learning_rate,num_epochs)))
        
        
    # make sure to plot the training curves
    if plot:
        plot_training_curve(train_losses, val_losses, None, None)
    return train_loss, val_loss



def testing(model, test_data, threshold: float = 10):
    model.eval()

    test_loss = 0.0
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()

    gt_list = []
    preds_list = []
    
    
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data

            outputs = model(inputs)
            # print(labels, outputs)
            gt_list.append(labels.tolist())
            preds_list.append(outputs.tolist())

            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_data)
    print(f'Test Loss: {test_loss}')

    # df = pd.DataFrame(columns=["G.Ts", "Preds.", "Diff.", "% Diff."])
    # df["G.Ts"] = gt_list
    # df["Preds."] = preds_list
    # df["Diff."] = 
    # percent_diffs = []
    
    for i in range(len(gt_list)):
        for j in range(len(gt_list[i])):
            GT = round(gt_list[i][j][0], 2)
            PD = round(preds_list[i][j][0], 2)
            DIFF = round(GT - PD, 2)
            PDIFF = round(abs(GT - PD) / GT * 100, 2)
            percent_diffs.append(PDIFF)
            
            print(f"GT: {GT:>5.2f}    PD: {PD:>5.2f}    DIFF: {DIFF:>5.2f}    %DIFF: {PDIFF:>5.2f}%")

    # Acceptable % difference <= 10%

    percent_diffs = np.array(percent_diffs)

    print(f"Percentage of inacceptable predictions: {round(len(percent_diffs[percent_diffs > threshold])/len(percent_diffs) * 100, 3)}%")
    print(f"Percentage of acceptable predictions: {round(len(percent_diffs[percent_diffs <= threshold])/len(percent_diffs) * 100, 3)}%")
    print(f"Average percentage difference for acceptable predictions: {round(percent_diffs[percent_diffs <= threshold].mean(), 3)}%")
    plt.title("Histogram of Percentage Differences Between G.Ts and Predictions")
    
    plt.grid(zorder=0)
    n, _, patches = plt.hist(percent_diffs, bins=100, zorder=5, color='g')
    
    for bin_value, patch in zip(n, patches):
        if bin_value > threshold:  
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('green')

    plt.ylim([0,max(n)])
    plt.vlines(x=10, ymin = 0, ymax = max(n), color='k', linestyles='--', zorder=6, label="Threshold for Accuracy")
    plt.xlabel("% Difference")
    plt.ylabel("Frequency")
    plt.legend(loc="best")