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
        self.train_loss = []
        self.validation_loss = []


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
                    average_loss = np.average(loss_values)
                    max = np.max(loss_values)
                    min = np.min(loss_values)
                    self.validation_loss.append((average_loss, max, min))
            else:
                print(f"警告: 文件 {file_path} 不存在。")

        for idx in range(self.start_idx, self.end_idx + 1):
            file_path = os.path.join(self.directory_path, f'training_loss_{idx}.txt')
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    # 假设每行是一个loss值，将其读取为浮点数并添加到loss_data列表中
                    loss_values = [float(line.strip()) for line in file.readlines()]
                    average_loss = np.average(loss_values)
                    max = np.max(loss_values)
                    min = np.min(loss_values)
                    self.train_loss.append((average_loss, max, min))
            else:
                print(f"警告: 文件 {file_path} 不存在。")

    def plot(self, epochs):
        """
        绘制训练损失随迭代次数变化的平均损失曲线图。
        :param title: 图表标题
        :param xlabel: x轴标签
        :param ylabel: y轴标签
        """
        # 模拟数据
        x = np.linspace(1,epochs,epochs)

        # 第一个折线的数据和误差
        train_loss = np.array([entry[0] for entry in self.train_loss])
        upper_loss = np.array([entry[1] for entry in self.train_loss])
        lower_loss = np.array([entry[2] for entry in self.train_loss])
        train_error = [upper_loss, lower_loss]  # 误差

        # 第二个折线的数据和误差
        eva_loss = np.array([entry[0] for entry in self.validation_loss])
        upper_loss = np.array([entry[1] for entry in self.validation_loss])
        lower_loss = np.array([entry[2] for entry in self.validation_loss])
        eva_error = [upper_loss, lower_loss]  # 误差

        # 创建图形
        plt.figure(figsize=(8, 6))

        # 绘制第一个带误差棒的折线图
        plt.errorbar(x, train_loss, yerr=train_error, fmt='-o', label='折线 1', capsize=5, color='blue')

        # 绘制第二个带误差棒的折线图
        plt.errorbar(x, eva_loss, yerr=eva_error, fmt='-s', label='折线 2', capsize=5, color='red')

        # 设置图表的标签和标题
        plt.xlabel('X 轴')
        plt.ylabel('Y 轴')
        plt.title('两个带误差棒的折线图')

        # 显示图例
        plt.legend()

        # 显示网格
        plt.grid(True)

        # 显示图形
        plt.show()



if __name__ == "__main__":
    plotter = LossPlotter(directory_path="loss", start_idx=0, end_idx=4)
    plotter.load_loss_data()
    print(plotter.validation_loss)
    plotter.plot(5)