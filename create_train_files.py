import os
import glob

# ========= 根据实际情况修改这里 =========
# EndoDAC 训练时使用的 data_path
DATA_PATH = "/home/hoo/X/depth/EndoDAC/data"

# 对应 scared_dataset 中的 folder 字段
# 也就是 data_path/train/<FOLDER>/data/left/...
FOLDER = "dataset1/keyframe_1"

# 输出的 splits 目录及文件名
SPLITS_DIR = "splits/endovis"
TRAIN_FILE_NAME = "train_files.txt"
# ======================================


def main():
    # left 图像所在目录
    left_dir = os.path.join(DATA_PATH, "train", FOLDER, "data", "left")
    if not os.path.isdir(left_dir):
        raise FileNotFoundError(f"left 目录不存在: {left_dir}")

    # 找到所有 frame_data*.png
    frame_paths = glob.glob(os.path.join(left_dir, "frame_data*.png"))
    frame_paths.sort()
    num_frames = len(frame_paths)

    if num_frames == 0:
        raise RuntimeError(f"在 {left_dir} 中没有找到 frame_data*.png")

    print(f"检测到 {num_frames} 张帧图像")

    # 准备 splits 输出目录
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_file_path = os.path.join(SPLITS_DIR, TRAIN_FILE_NAME)

    # 写 train_files.txt：只用左相机（l），frame_index 从 1 开始
    with open(train_file_path, "w") as f:
        for idx in range(num_frames):
            frame_index = idx + 1  # 1..N，对应磁盘上的 frame_data000000..(N-1)
            line = f"{FOLDER}\t{frame_index}\t l\n"
            f.write(line)

    print(f"train_files 写入完成: {train_file_path}")


if __name__ == "__main__":
    main()
