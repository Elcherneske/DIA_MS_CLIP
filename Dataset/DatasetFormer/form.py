import argparse
import pickle
import shutil

from FraggerPipeFormer import DataFormer, DataConsolation
from Preprocessor import Preprocessor
import os
from pathlib import Path

def init_arg_parser():
    parser = argparse.ArgumentParser()
    #filename parameters
    parser.add_argument('--raw_path', default='./data/raw')
    parser.add_argument('--mid_path', default='./data/mid')
    parser.add_argument('--out_filename', default='./data.pkl')
    return parser

# def main():
#     parser = init_arg_parser()
#     args = parser.parse_args()
#
#     dataformer = DataFormer(posPreFilename=args.pos_precursor_file, posFragFilename=args.pos_fragment_file, negPreFilename=args.neg_precursor_file, negFragFilename=args.neg_fragment_file)
#     dataformer.dump_all('data.pkl')
#     preprocessor = Preprocessor('data.pkl')
#     preprocessor.preprocess(RT_dim=50, outputFilename=args.out_filename, filter_options=[{'mode': 'ion_num', 'value': 10}])


def main():
    # 文件夹路径
    parser = init_arg_parser()
    args = parser.parse_args()
    pos_raw_path = os.path.join(args.raw_path, 'pos')
    neg_raw_path = os.path.join(args.raw_path, 'neg')
    mid_pos_path = os.path.join(args.mid_path, 'pos')
    mid_neg_path = os.path.join(args.mid_path, 'neg')

    if Path(args.mid_path).exists():
        # 使用 shutil.rmtree 删除非空文件夹及其所有内容
        for item in Path(args.mid_path).iterdir():
            if item.is_dir():
                shutil.rmtree(item)  # 递归删除文件夹及其中内容
            else:
                item.unlink()  # 删除文件

    # 确保中间结果目录存在
    Path(mid_pos_path).mkdir(parents=True, exist_ok=True)
    Path(mid_neg_path).mkdir(parents=True, exist_ok=True)

    # 遍历 ./data/raw/pos 文件夹中的所有子文件夹，处理正样本
    for folder in os.listdir(pos_raw_path):
        folder_path = os.path.join(pos_raw_path, folder)
        if os.path.isdir(folder_path):  # 确保是一个文件夹
            pre_filename = os.path.join(folder_path, 'pre.pkl')  # 预处理文件路径
            frag_filename = os.path.join(folder_path, 'frag.pkl')  # 片段文件路径
            if os.path.exists(pre_filename) and os.path.exists(frag_filename):
                # 使用 DataFormer 处理该文件夹的数据
                data_former = DataFormer(pre_filename, frag_filename, filter_label=1)
                out_filename = os.path.join(mid_pos_path, f"{folder}.pkl")
                data_former.dump(out_filename, shuffle=True)
            else:
                print(f"Warning: {folder} 中缺少 pre 或 frag 文件。")

    # 遍历 ./data/raw/neg 文件夹中的所有子文件夹，处理负样本
    for folder in os.listdir(neg_raw_path):
        folder_path = os.path.join(neg_raw_path, folder)
        if os.path.isdir(folder_path):  # 确保是一个文件夹
            pre_filename = os.path.join(folder_path, 'pre.pkl')  # 预处理文件路径
            frag_filename = os.path.join(folder_path, 'frag.pkl')  # 片段文件路径
            if os.path.exists(pre_filename) and os.path.exists(frag_filename):
                # 使用 DataFormer 处理该文件夹的数据
                data_former = DataFormer(pre_filename, frag_filename, filter_label=-1)
                out_filename = os.path.join(mid_neg_path, f"{folder}.pkl")
                data_former.dump(out_filename, shuffle=True)
            else:
                print(f"Warning: {folder} 中缺少 pre 或 frag 文件。")

    # 使用 DataConsolation 合并正负样本
    data_consolation = DataConsolation(pos_path=mid_pos_path, neg_path=mid_neg_path)
    # 检查正负样本标签是否符合要求
    pos_check, neg_check = data_consolation.check()
    if pos_check and neg_check:
        # 合并正负样本
        combined_data = data_consolation.dump()
        # 保存合并后的数据
        mid_filename = os.path.join(args.mid_path, 'tmp.pkl')
        with open(mid_filename, 'wb') as f:
            pickle.dump(combined_data, f)
        print(f"合并后的数据已保存到 {args.out_filename}，样本数量：{len(combined_data)},开始预处理")

        preprocessor = Preprocessor(mid_filename)
        preprocessor.preprocess(RT_dim=50, outputFilename=args.out_filename, filter_options=[{'mode': 'ion_num', 'value': 10}])

    else:
        print("正负样本标签检查未通过，合并数据失败。")

if __name__ == "__main__":
    main()
