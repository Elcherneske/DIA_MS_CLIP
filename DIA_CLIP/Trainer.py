import argparse
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
import time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from tqdm import tqdm
from Loss import *
from Visualizer import Visualizer


class ModelTrainer():
    def __init__(self, model, criterion, optimizer='adam', lr=0.0001, is_save_model = True, is_model_init = True):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.vis = Visualizer()
        self.is_save_model = is_save_model
        self.is_model_init = is_model_init
        if optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = SGD(self.model.parameters(), lr=lr)

    def train(self, train_loader, num_epochs, retrain_path=None):
        if retrain_path:
            self.model.load(retrain_path)
        else:
            if self.is_model_init:
                self.model.parameter_init()
                print("No retrain, model init")
            else:
                print("No retrain, model default init")

        for epoch in range(num_epochs):
            self.train_one_epoch(epoch, train_loader)
            if self.is_save_model:
                self.save_model(epoch)

    def train_and_validation(self, train_loader, val_loader, num_epochs, retrain_path=None):
        if retrain_path:
            self.model.load(retrain_path)
        else:
            if self.is_model_init:
                self.model.parameter_init()
                print("No retrain, model init")
            else:
                print("No retrain, model default init")

        for epoch in range(num_epochs):
            self.train_one_epoch(epoch=epoch, num_epochs=num_epochs, train_loader=train_loader)
            if self.is_save_model:
                self.save_model(epoch)
            self.validate(val_loader=val_loader, epoch=epoch, plot_hotfig=True)

    def train_one_epoch(self, train_loader, epoch, num_epochs):
        self.model.train()
        loss_epoch = []
        with tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{num_epochs}") as pbar:
            for n_step, data in enumerate(pbar):
                peptide_feature, chrom_feature = self.model(data)
                self.optimizer.zero_grad()
                loss_dict = self.criterion(peptide_feature, chrom_feature, data["label"])
                loss_dict['loss'].backward()
                self.optimizer.step()
                loss_epoch.append(loss_dict['loss'].item())

                if n_step % 25 == 0:
                    pbar.set_postfix(n_step=n_step, Loss=loss_dict['loss'].item(), Loss_t=loss_dict['loss_t'].item(), Loss_f=loss_dict['loss_f'].item(), Inner_Loss=loss_dict['inner_loss'].item())
                if n_step % 1000 == 0:
                    self.vis.plotHotMap(peptide_feature=peptide_feature, chrom_feature=chrom_feature, label=data["label"], fig_name="train_epoch_" + str(epoch) + "_n_" + str(n_step) + '_heatmap_imshow.png')

        print(f'Epoch [{epoch + 1}], Ave_Loss: {np.sum(loss_epoch) / len(train_loader):.4f}')

        if not os.path.exists('./loss'):
            os.makedirs('./loss')
            print(f"路径 {'./loss'} 已创建")
        with open("./loss/training_loss_" + str(epoch) + ".txt", "w") as fp:
            for loss in loss_epoch:
                fp.write(str(loss) + "\n")

    def save_model(self, epoch_ID):
        if not os.path.exists('model'):
            os.makedirs('model')
            print(f"路径 {'./model'} 已创建")
        torch.save(self.model.state_dict(), "./model/epoch_" + str(epoch_ID) + "_model.pt")
        print(f'Model saved to {"./model/epoch_" + str(epoch_ID) + "_model.pt"}')

    def validate(self, val_loader, model_path=None, epoch=0, plot_hotfig=True):
        if not val_loader:
            print("No validation")
            return
        if model_path:
            self.model.load(model_path)
        self.model.eval()  # 设置模型为评估模式
        print("Start validation")
        loss_epoch = []
        label = []
        score = []
        with torch.no_grad():
            with tqdm(val_loader, desc=f"Validation:") as pbar:
                for n_step, data in enumerate(pbar):
                    peptide_feature, chrom_feature = self.model(data)
                    loss_dict = self.criterion(peptide_feature, chrom_feature, data["label"])
                    loss_epoch.append(loss_dict['loss'].item())
                    label += data["label"].cpu().detach().numpy().tolist()
                    score += (-1 * torch.norm(peptide_feature - chrom_feature, p=2, dim=-1, keepdim=False)).cpu().detach().numpy().tolist()
                    if plot_hotfig and n_step % 100 == 0:
                        pbar.set_postfix(n_step=n_step, Loss=loss_dict['loss'].item(), Loss_t=loss_dict['loss_t'].item(), Loss_f=loss_dict['loss_f'].item(), Inner_Loss=loss_dict['inner_loss'].item())
                        self.vis.plotHotMap(peptide_feature=peptide_feature, chrom_feature=chrom_feature, label=data["label"], fig_name="val_epoch_" + str(epoch) + "_n_" + str(n_step) + '_heatmap_imshow.png')

            # calculate AUC and AUPRC
            AUC = roc_auc_score(label, score)
            print(f"AUC: {AUC}")
            fpr, tpr, thresholds = roc_curve(label, score)
            precision, recall, thresholds = precision_recall_curve(label, score)
            AUPRC = auc(recall, precision)
            print(f"AUPRC: {AUPRC}")
            if plot_hotfig:
                self.vis.plotROC(fpr=fpr, tpr=tpr, auc=AUC, fig_name="val_epoch_" + str(epoch) + '_ROC_imshow.png')
                self.vis.plotPR(recall=recall, precision=precision, auprc=AUPRC, fig_name="val_epoch_" + str(epoch) + '_PR_imshow.png')

            # output loss to .txt
            if not os.path.exists('./loss'):
                os.makedirs('./loss')
                print(f"路径 {'./loss'} 已创建")
            with open("./loss/evaluation_loss.txt", "w") as fp:
                for loss in loss_epoch:
                    fp.write(str(loss) + "\n")

        print(f'Validation Average Loss: {np.sum(loss_epoch) / len(val_loader):.4f}')

    def UMAP_eva(self, train_loader, test_loader, model_path=None, check_mode=0, item_mode=0):
        # check_mode = 0  #0-label     1-train/test
        # item_mode = 0   #0-peptide   1-chrom
        print(f"UMAP mode: check mode:{check_mode}, item mode:{item_mode}")
        if model_path:
            self.model.load(model_path)
        self.model.eval()
        feature_list = []
        label_list = []
        with torch.no_grad():
            for data in train_loader:
                peptide_feature, chrom_feature = self.model(data)
                peptide_feature = peptide_feature.cpu().detach().numpy()
                chrom_feature = chrom_feature.cpu().detach().numpy()
                label = data["label"].cpu().detach().numpy()
                if check_mode == 0:
                    for index in range(peptide_feature.shape[0]):
                        label_list.append(label[index])
                        if item_mode == 0:
                            feature_list.append(peptide_feature[index])
                        elif item_mode == 1:
                            feature_list.append(chrom_feature[index])
                if check_mode == 1:
                    for index in range(peptide_feature.shape[0]):
                        label_list.append(1.0)
                        if item_mode == 0:
                            feature_list.append(peptide_feature[index])
                        elif item_mode == 1:
                            feature_list.append(chrom_feature[index])

            for data in test_loader:
                peptide_feature, chrom_feature = self.model(data)
                peptide_feature = peptide_feature.cpu().detach().numpy()
                chrom_feature = chrom_feature.cpu().detach().numpy()
                label = data["label"].cpu().detach().numpy()
                if check_mode == 0:
                    for index in range(peptide_feature.shape[0]):
                        label_list.append(label[index])
                        if item_mode == 0:
                            feature_list.append(peptide_feature[index])
                        elif item_mode == 1:
                            feature_list.append(chrom_feature[index])
                if check_mode == 1:
                    for index in range(peptide_feature.shape[0]):
                        label_list.append(0.0)
                        if item_mode == 0:
                            feature_list.append(peptide_feature[index])
                        elif item_mode == 1:
                            feature_list.append(chrom_feature[index])

        self.vis.UmapPlot(feature=feature_list, label=label_list, fig_name=f"UMAP_eva_{check_mode}_{item_mode}")


if __name__ == "__main__":
    a = []
    x = torch.tensor([1,2,3])
    a += x.detach().numpy().tolist()
    print(a)
