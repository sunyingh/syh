import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# 根目录 = 当前文件所在目录
ROOT_DIR = os.path.dirname(__file__)
# npz 所在目录：splits/endovis/curve
CURVE_DIR = os.path.join(ROOT_DIR, "splits", "endovis", "curve")

def load_poses_npz(path):
    data = np.load(path)
    poses = data["data"]      # 如果 keys 不是 data，下面说明怎么改
    return poses

def mat4_to_t_quat(T):
    R_mat = T[:3, :3]
    t = T[:3, 3]
    r = R.from_matrix(R_mat)
    qx, qy, qz, qw = r.as_quat()
    return t, (qx, qy, qz, qw)

def export_one(npz_path, txt_path):
    poses = load_poses_npz(npz_path)
    print(f"Loaded {poses.shape[0]} poses from {npz_path}")

    with open(txt_path, "w") as f:
        f.write("# idx tx ty tz qx qy qz qw\n")
        for i, T in enumerate(poses):
            t, q = mat4_to_t_quat(T)
            line = "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                i, t[0], t[1], t[2], q[0], q[1], q[2], q[3]
            )
            f.write(line)
    print(f"Saved poses (t + quat) to {txt_path}")

def main():
    npz_1 = os.path.join(CURVE_DIR, "pred_poses_sequence1.npz")
    npz_2 = os.path.join(CURVE_DIR, "pred_poses_sequence2.npz")

    txt_1 = os.path.join(CURVE_DIR, "pred_poses_sequence1_quat.txt")
    txt_2 = os.path.join(CURVE_DIR, "pred_poses_sequence2_quat.txt")

    export_one(npz_1, txt_1)
    export_one(npz_2, txt_2)

if __name__ == "__main__":
    main()
