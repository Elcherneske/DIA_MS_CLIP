import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
import time
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from Trainer import ModelTrainer
from MSDataset import MSDataset, DataChecker
from DIAClip import DIAClip, custom_model_init
from Loss import *
from Visualizer import Visualizer




def init_arg_parser():
    parser = argparse.ArgumentParser()
    #database parameters
    parser.add_argument('--file_path', default='../dataset/')
    parser.add_argument('--mz_divide_charge', action='store_true', default=False)
    parser.add_argument('--chrom_channel_norm', action='store_true', default=False)
    parser.add_argument('--dataset', default='ms')
    #model parameters
    parser.add_argument('--RT_dim', default=50, type=int)
    parser.add_argument('--hidden_layer', default=2048, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--n_head', default=64, type=int)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--max_mz_range', default=5000, type=float)
    parser.add_argument('--bin_size', default=2500, type=int)
    parser.add_argument("--peptide_vocab_path", default="./peptide_vocab.txt")
    parser.add_argument("--modification_vocab_path", default="./modification_vocab.txt")
    parser.add_argument('--model_peptide_mode', default='trans', choices=["trans", "esm2"], type=str)
    parser.add_argument('--model_spec_mode', default='spec_split_RT', choices=["chrom", "chrom_split", "spec", "spec_RT", "spec_split_RT"], type=str)
    #train parameters
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--retrain_path', action="store", type=str, help='specify a saved model to retrain on')
    parser.add_argument("--optimizer", default="adam")
    parser.add_argument("--eva_model_path", default=None)
    parser.add_argument("--is_save_train_model", action='store_true', default=True)
    parser.add_argument("--is_model_init", action='store_true', default=False)
    parser.add_argument("--mode", choices=["eva", "train", "train_eva", "UMAP_vis"], default="train_eva")
    #loss parameters
    parser.add_argument('--loss_function', default='CrossEntropyLoss', type=str)
    parser.add_argument('--loss_alpha', default=0.1, type=float)
    parser.add_argument('--loss_beta', default=0.01, type=float)
    parser.add_argument('--loss_gamma', default=0.01, type=float)
    parser.add_argument('--loss_delta', default=0.05, type=float)
    parser.add_argument('--loss_only_true', action='store_true', default=False)
    parser.add_argument('--loss_is_dict', action='store_true', default=True)
    parser.add_argument('--loss_mode', default=1, type=int)
    parser.add_argument('--loss_consider_inner_dist', action='store_true', default=True)
    #visualize parameters
    parser.add_argument('--UMAP_check_mode', default=0, type=int)
    parser.add_argument('--UMAP_item_mode', default=0, type=int)
    return parser



def main():
    parser = init_arg_parser()
    args = parser.parse_args()
    device = args.device

    if args.dataset == 'ms':
        data_checker = DataChecker(RT_dim=args.RT_dim, ion_num=10)
        train_dataset = MSDataset(file_path=args.file_path, data_checker=data_checker, is_train=True, device=device, divide_charge=args.mz_divide_charge, channel_norm=args.chrom_channel_norm)
        test_data = MSDataset(file_path=args.file_path, data_checker=data_checker, is_train=False, device=device, divide_charge=args.mz_divide_charge, channel_norm=args.chrom_channel_norm)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"Dataset: file_path: {args.file_path}, divide_charge:{args.mz_divide_charge}, channel_norm:{args.chrom_channel_norm}")
    print(f"Train Dataset Length: {len(train_dataset)}; Validation Dataset Length: {len(test_data)}")

    dia_clip = DIAClip(d_model=args.d_model, RT_dim=args.RT_dim, n_head=args.n_head, device=device, dim_feedforward=args.hidden_layer, dropout=args.dropout, peptide_vocab=args.peptide_vocab_path, modification_vocab=args.modification_vocab_path, mode_peptide=args.model_peptide_mode, mode_spec=args.model_spec_mode).to(device)
    print(f"Model Params: d_model:{args.d_model}, RT_dim:{args.RT_dim}, hidden_layer: {args.hidden_layer}, n_head: {args.n_head}, dropout: {args.dropout}")
    print(f"parameter number: {sum(p.numel() for p in dia_clip.parameters())}")

    trainer = ModelTrainer(model=dia_clip, optimizer=args.optimizer, lr=args.lr, is_save_model=args.is_save_train_model, is_model_init=args.is_model_init)
    if args.loss_function == "InfoNCELoss":
        loss_fn = InfoNCELoss(exp_ratio=args.loss_exp_ratio)
    elif args.loss_function == "CrossEntropyLoss":
        loss_fn = CrossEntropyLoss(beta=args.loss_beta, alpha=args.loss_alpha, gamma=args.loss_gamma, only_true=args.loss_only_true, is_dist=args.loss_is_dict, loss_mode=args.loss_mode, delta=args.loss_delta, consider_inner_dist=args.loss_consider_inner_dist)
        print(f"loss function: {args.loss_function}, beta:{args.loss_beta}, alpha:{args.loss_alpha}, gamma:{args.loss_gamma}, delta:{args.loss_delta}, only_true:{args.loss_only_true}, is_dict:{args.loss_is_dict}, loss_mode:{args.loss_mode}, consider_inner_dist:{args.loss_consider_inner_dist}")

    print(f"Trainner Params: epochs: {args.epoch}, batch: {args.batch_size}, lr: {args.lr}, device: {args.device}, mode:{args.mode}")
    if args.mode == "train":
        trainer.train(criterion=loss_fn, train_loader=train_loader, num_epochs=args.epoch, retrain_path=args.retrain_path)
        trainer.validate(criterion=loss_fn, val_loader=test_loader, model_path=args.eva_model_path, plot_hotfig=True)
    elif args.mode == "eva":
        trainer.validate(criterion=loss_fn, val_loader=test_loader, model_path=args.eva_model_path, plot_hotfig=True)
    elif args.mode == "train_eva":
        trainer.train_and_validation(criterion=loss_fn, train_loader=train_loader, val_loader=test_loader, num_epochs=args.epoch, retrain_path=args.retrain_path)
    elif args.mode == "UMAP_vis":
        trainer.UMAP_eva(train_loader=train_loader, test_loader=test_loader, model_path=args.eva_model_path, check_mode=args.UMAP_check_mode, item_mode=args.UMAP_item_mode)



if __name__ == "__main__":
    main()