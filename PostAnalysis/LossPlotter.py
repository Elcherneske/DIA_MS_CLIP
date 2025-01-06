import os
import matplotlib.pyplot as plt
import numpy as np

class LossPlotter:
    def __init__(self, directory_path, start_idx, end_idx):
        """
        初始化 LossPlotter 类，输入路径以及编号范围。
        :param directory_path: 存放txt文件的目录路径
        :param start_idx: 读取文件的起始编号
        :param end_idx: 读取文件的结束编号
        """
        self.directory_path = directory_path
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.loss_data = []

    def load_loss_data(self):
        """
        读取指定编号范围内的 training_loss_x.txt 文件，并将其中的 loss 数据加载到列表中。
        """
        for idx in range(self.start_idx, self.end_idx + 1):
            file_path = os.path.join(self.directory_path, f'evaluation_{idx}_loss.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    # 假设每行是一个loss值，将其读取为浮点数并添加到loss_data列表中
                    loss_values = [float(line.strip()) for line in file.readlines()]
                    self.loss_data.extend(loss_values)
            else:
                print(f"警告: 文件 {file_path} 不存在。")

    def calculate_average_loss(self):
        """
        计算每个迭代步骤的平均损失（即对每个迭代步骤，在所有文件中的损失值求平均）。
        :return: 每个迭代步骤的平均损失
        """
        if not self.loss_data:
            print("没有加载到损失数据，请先加载数据。")
            return []

        avg_loss_per_iteration = []
        for i in range(len(self.loss_data)):
            length = i + 1
            avg_loss_per_iteration.append(np.average(self.loss_data[:length]))
        return avg_loss_per_iteration

    def plot_loss(self, title="Training Loss Curve", xlabel="Iteration", ylabel="Loss"):
        """
        绘制训练损失随迭代次数变化的曲线图。
        :param title: 图表标题
        :param xlabel: x轴标签
        :param ylabel: y轴标签
        """
        if not self.loss_data:
            print("没有加载到损失数据，请先加载数据。")
            return

        # 生成迭代次数
        iterations = list(range(1, len(self.loss_data) + 1))

        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, self.loss_data, label="Training Loss", color="blue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def plot_average_loss(self, title="Average Training Loss Curve", xlabel="Iteration", ylabel="Average Loss"):
        """
        绘制训练损失随迭代次数变化的平均损失曲线图。
        :param title: 图表标题
        :param xlabel: x轴标签
        :param ylabel: y轴标签
        """
        avg_loss = self.calculate_average_loss()

        if not avg_loss:
            print("无法计算平均损失，请确保已经加载了损失数据。")
            return

        # 生成迭代次数
        iterations = list(range(1, len(avg_loss) + 1))

        # 绘制平均损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, avg_loss, label="Average Training Loss", color="blue")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.show()



if __name__ == "__main__":
    plotter = LossPlotter(directory_path="loss", start_idx=0, end_idx=4)
    plotter.load_loss_data()
    plotter.plot_average_loss()