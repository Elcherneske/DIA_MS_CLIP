import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
from MSDataset import MSDataset
import os

class DataVisualizer():
    def __init__(self):
        current_directory = os.getcwd()

        # 判断文件夹是否存在
        if os.path.isdir(os.path.join(current_directory, "fig")):
            print(f"文件夹 'fig' 存在")
        else:
            print(f"文件夹 'fig' 不存在")

    def pearson_correlation(self, x, y):
        """
        :param x: [N, dim]
        :param y: [N, dim]
        :return: [N]
        """
        # 计算均值
        mean_x = torch.mean(x, dim=-1)
        mean_y = torch.mean(y, dim=-1)

        # 计算分子和分母
        numerator = torch.sum((x - mean_x.unsqueeze(-1)) * (y - mean_y.unsqueeze(-1)), dim=-1)
        denominator = torch.sqrt(
            torch.sum((x - mean_x.unsqueeze(-1)) ** 2, dim=-1) * torch.sum((y - mean_y.unsqueeze(-1)) ** 2, dim=-1))

        return numerator / (denominator + 1e-6)

    def vis_chrom(self, data, id):
        peptide_chrom = data["peptide_chrom"].detach().cpu().numpy()
        peptide_mz = data["peptide_mz"].detach().cpu().numpy()
        peptide_RT = data["peptide_RT"].detach().cpu().numpy()
        fragment_chrom = data["fragment_chrom"].detach().cpu().numpy()
        fragment_mz = data["fragment_mz"].detach().cpu().numpy()
        fragment_RT = data["fragment_RT"].detach().cpu().numpy()
        label = data["label"].detach().cpu().numpy()
        print(label)

        fig, axs = plt.subplots(1, 2)
        for index, chrom in enumerate(peptide_chrom):
            axs[0].plot(peptide_RT, chrom, label=peptide_mz[index])
        axs[0].set_title("precursors chromatogram")  # 设置标题
        axs[0].legend(fontsize=6)
        axs[0].set_xlabel('Retention Time')  # 设置横坐标标题
        axs[0].set_ylabel('Intensity')  # 设置纵坐标标题

        for index, chrom in enumerate(fragment_chrom):
            axs[1].plot(fragment_RT, chrom, label=fragment_mz[index])
        axs[1].set_title("fragments chromatogram")  # 设置标题
        axs[1].legend(fontsize=6)
        axs[1].set_xlabel('Retention Time')  # 设置横坐标标题
        axs[1].set_ylabel('Intensity')  # 设置纵坐标标题

        # 显示图形
        plt.tight_layout()
        if label == 0:
            plt.savefig("./fig/negtive/" + "negtive_" + str(id) + "_fig.png", dpi=300)  # dpi 参数控制图片的分辨率
        elif label == 1:
            plt.savefig("./fig/positive/" + "positive_" + str(id) + "_fig.png", dpi=300)  # dpi 参数控制图片的分辨率
        else:
            plt.savefig("./fig/unknown/" + "unknown_" + str(id) + "_fig.png", dpi=300)  # dpi 参数控制图片的分辨率
        plt.close()


def init_arg_parser():
    parser = argparse.ArgumentParser()
    #database parameters
    parser.add_argument('--data_file_path', default='./data.pkl')
    return parser


def main():
    parser = init_arg_parser()
    args = parser.parse_args()
    dataset = MSDataset(filename = args.data_file_path, is_train=True, train_ratio=1.0, divide_charge=False, channel_norm=False)
    visualizer = DataVisualizer()
    print(f"dataset len: {len(dataset)}")
    for i in range(0, 100):
        visualizer.vis_chrom(data=dataset[i], id=i)



if __name__ == "__main__":
    main()
















