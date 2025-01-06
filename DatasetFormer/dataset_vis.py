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

    def vis_spec_2D(self, data, id):
        peptide_chrom = data["peptide_chrom"]
        peptide_mz = data["peptide_mz"]
        peptide_RT = data["peptide_RT"]
        fragment_chrom = data["fragment_chrom"]
        fragment_mz = data["fragment_mz"]
        fragment_RT = data["fragment_RT"]
        label = data["label"]
        print(label)

        max_bin_mz = 2500
        bin_size = 100

        spec = torch.cat([peptide_chrom, fragment_chrom], dim=0)
        spec = spec.unsqueeze(0)
        mz = torch.cat([peptide_mz, fragment_mz], dim=0)
        mz = mz.unsqueeze(0)
        spec_feature = self.spec_bin_emb(spec=spec, mz=mz, max_bin_mz=max_bin_mz, bin_size=bin_size).squeeze(0) #[RT_dim, bin_size]
        RT_dim, _ = spec_feature.shape
        mz = torch.arange(0, bin_size, 1) * (max_bin_mz / bin_size) #[bin_size]
        mz = mz.unsqueeze(0).repeat(RT_dim, 1)
        RT = peptide_RT.unsqueeze(-1).repeat(1, bin_size)

        plt.scatter(
            x=RT.reshape(-1).detach().cpu().numpy(),
            y=mz.reshape(-1).detach().cpu().numpy(),
            c=spec_feature.reshape(-1).detach().cpu().numpy(),
            cmap="afmhot_r",
            s=10,
            alpha=1.0
        )
        plt.xlabel("time (s)")
        plt.ylabel("m/z")
        plt.colorbar()  # 显示value对应的颜色的bar
        # 显示图形
        # plt.show()
        plt.tight_layout()

        if label == 0:
            plt.savefig("./fig/negtive/" + "negtive_" + str(id) + "_fig.png", dpi=300)  # dpi 参数控制图片的分辨率
        elif label == 1:
            plt.savefig("./fig/positive/" + "positive_" + str(id) + "_fig.png", dpi=300)  # dpi 参数控制图片的分辨率
        else:
            plt.savefig("./fig/unknown/" + "unknown_" + str(id) + "_fig.png", dpi=300)  # dpi 参数控制图片的分辨率
        plt.close()


    def vis_3D(self, data, id):
        peptide_chrom = data["peptide_chrom"].detach().cpu().numpy()
        peptide_mz = data["peptide_mz"].detach().cpu().numpy()
        peptide_RT = data["peptide_RT"].detach().cpu().numpy()
        fragment_chrom = data["fragment_chrom"].detach().cpu().numpy()
        fragment_mz = data["fragment_mz"].detach().cpu().numpy()
        fragment_RT = data["fragment_RT"].detach().cpu().numpy()
        label = data["label"].detach().cpu().numpy()
        print(label)

        # 创建一个 3D 图形
        fig = plt.figure(figsize=(12, 6))

        peptide_RT, peptide_mz = np.meshgrid(peptide_RT, peptide_mz)
        fragment_RT, fragment_mz = np.meshgrid(fragment_RT, fragment_mz)

        # 绘制三维散点图
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(peptide_RT, peptide_mz, peptide_chrom, c='r', marker='o')
        ax1.set_title('3D Surface Plot')
        ax1.set_xlabel('RT axis')
        ax1.set_ylabel('mz axis')
        ax1.set_zlabel('Z axis')

        # 绘制三维表面图
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(fragment_RT, fragment_mz, fragment_chrom, c='r', marker='o')
        ax2.set_title('3D Surface Plot')
        ax2.set_xlabel('RT axis')
        ax2.set_ylabel('mz axis')
        ax2.set_zlabel('Z axis')

        # 显示图形
        plt.tight_layout()
        if label == 0:
            plt.savefig("./fig/negtive/" + "negtive_" + str(id) +"_fig.png", dpi=300)  # dpi 参数控制图片的分辨率
        else:
            plt.savefig("./fig/positive/" + "positive_" + str(id) +"_fig.png", dpi=300)  # dpi 参数控制图片的分辨率
        plt.close()


    def spec_bin_emb(self, spec, mz, max_bin_mz=5000, bin_size=2500, add_noise=False):
        """
        :param spec: [batch, ion_num, RT_dim]
        :param mz: [batch, ion_num]
        :param max_bin_mz: int/float
        :return: spec_feature: [batch, RT_dim, bin_size]
        """
        batch, ion_num, RT_dim = spec.shape
        device = spec.device
        target = torch.zeros(batch, RT_dim, bin_size).to(device)
        noise = torch.rand(batch, RT_dim, bin_size).to(device)
        indices = (mz / max_bin_mz * bin_size).long()  # [B, ion_num]
        indices = indices.unsqueeze(1).repeat(1, RT_dim, 1)  # [B, RT_dim, ion_num]
        indices = torch.clamp(indices, 0, bin_size - 1) # 确保索引在有效范围内
        spec = spec.permute(0,2,1) #[batch, RT_dim, ion_num]
        target = target.scatter_add(dim = -1, index = indices, src = spec)
        if add_noise:
            target += noise * 0.1
        return target


    def check_chrom_same(self, chrom1, chrom2):
        if not (len(chrom1) == len(chrom2)):
            return False
        if not (len(chrom1[0]) == len(chrom2[0])):
            return False

        for index in range(len(chrom1)):
            if not (np.all(chrom1[index] == chrom2[index])):
                return False
        return True

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
















