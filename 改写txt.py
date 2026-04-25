'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2026-01-28 11:31:27
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2026-01-28 11:34:39
FilePath: \EndoDAC\改写txt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os

# 需要处理的 txt 文件列表（可以按需增减）
SPLIT_FILES = [
    r"./splits/endovis_blur/train_files.txt",
    r"./splits/endovis_blur/val_files.txt",
    r"./splits/endovis_blur/test_files.txt",
     # 如果也想一起改，可以加上
]

def process_line(line: str) -> str:
    """
    输入一行，例如: 'dataset2/keyframe3\t5\tl'
    根据规则决定是否把第三列的 'l' 改成 'left_blur'。
    """

    parts = line.strip().split()
    if len(parts) != 3:
        # 不符合三列格式的行原样返回
        return line

    folder, frame_idx_str, token = parts

    # 只处理原本是 'l' 的
    if token != "l":
        return line

    # 解析 datasetX 和 keyframeX
    # folder 形如 'dataset2/keyframe_3'

    try:
        dataset_part, keyframe_part = folder.split("/")
    except ValueError:
        # 格式不符合预期，直接跳过
        return line

    # dataset 编号
    if not dataset_part.startswith("dataset"):
        return line
    try:
        dataset_id = int(dataset_part.replace("dataset", ""))
    except ValueError:
        return line

    # keyframe 编号，命名格式 keyframe_x
    if not keyframe_part.startswith("keyframe_"):
        return line
    try:
        keyframe_id = int(keyframe_part.replace("keyframe_", ""))
    except ValueError:
        return line

    # 规则 1：
    # dataset1、dataset2、dataset3 的 keyframe1-4 的第三列 'l' -> 'left_blur'
    if 1 <= dataset_id <= 3 and 1 <= keyframe_id <= 4:
        token = "left_blur"

    # 规则 2：
    # dataset4-9 的 keyframe3-4 的第三列 'l' -> 'left_blur'
    if 4 <= dataset_id <= 9 and 3 <= keyframe_id <= 4:
        token = "left_blur"

    # 用单个空格重新拼接（原来是 \t 也没关系，split+join 会归一化）
    return f"{folder}\t{frame_idx_str}\t{token}\n"


def process_file(path: str):
    print(f"Processing {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = [process_line(line) for line in lines]

    # 覆盖写回原文件前，可以先备份一份
    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  Backup saved to {backup_path}")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print(f"  Done.")


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    for rel_path in SPLIT_FILES:
        full_path = os.path.join(base_dir, rel_path)
        if os.path.exists(full_path):
            process_file(full_path)
        else:
            print(f"Skip (not found): {full_path}")
