import argparse
import os
import shutil
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--split_file",
        type=str,
        default=os.path.join("splits", "endovis", "test_files.txt"),
    )
    p.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="SCARED root or train folder (e.g. /media/.../SCARED or /media/.../SCARED/train)",
    )
    p.add_argument(
        "--dst_root",
        type=str,
        required=True,
        help="Destination root to copy extracted tiff depths into",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only produce reports without copying",
    )
    return p.parse_args()


def normalize_folder_token(folder_token: str) -> str:
    return folder_token.strip().replace("\\", "/")


def parse_keyframe_num(folder_token: str) -> str:
    folder_token = normalize_folder_token(folder_token)
    parts = folder_token.split("/")
    if len(parts) != 2:
        return ""
    keyframe = parts[1]
    if keyframe.startswith("keyframe_"):
        return keyframe[len("keyframe_") :]
    if keyframe.startswith("keyframe"):
        return keyframe[len("keyframe") :]
    return ""


def build_src_candidates(src_root: str, folder_token: str, frame_index: int) -> list:
    folder_token = normalize_folder_token(folder_token)
    file_id = frame_index - 1
    filename = f"scene_points{file_id:06d}.tiff"

    roots = [src_root, os.path.join(src_root, "train"), os.path.join(src_root, "test")]

    cands = []
    for root in roots:
        cands.append(os.path.join(root, folder_token, "data", "scene_points", filename))

    return cands


def resolve_src_path(src_root: str, folder_token: str, frame_index: int) -> str:
    cands = build_src_candidates(src_root, folder_token, frame_index)
    for p in cands:
        if os.path.exists(p):
            return p
    return cands[0]


def build_dst_path(dst_root: str, folder_token: str, frame_index: int, side_token: str) -> str:
    folder_token = normalize_folder_token(folder_token)
    parts = folder_token.split("/")
    dataset = parts[0] if parts else folder_token
    keyframe_tag = parts[1] if len(parts) >= 2 else ""

    file_id = frame_index - 1
    side_token = (side_token or "left").strip()

    # Flat output: all files under dst_root
    if keyframe_tag:
        out_name = f"{dataset}_{keyframe_tag}_{side_token}_frame_data{file_id:06d}.tiff"
    else:
        out_name = f"{dataset}_{side_token}_frame_data{file_id:06d}.tiff"
    return os.path.join(dst_root, out_name)


def main():
    args = parse_args()

    split_file = args.split_file
    if not os.path.isabs(split_file):
        split_file = os.path.abspath(split_file)

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"split_file not found: {split_file}")

    os.makedirs(args.dst_root, exist_ok=True)

    missing = []
    copied = 0

    by_src_path = defaultdict(list)

    with open(split_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            folder_token = parts[0]
            frame_index = int(parts[1])
            side_token = parts[2] if len(parts) >= 3 else "left"

            src_path = resolve_src_path(args.src_root, folder_token, frame_index)
            dst_path = build_dst_path(args.dst_root, folder_token, frame_index, side_token)

            by_src_path[src_path].append((line_no, folder_token, frame_index, side_token))

            if not os.path.exists(src_path):
                missing.append((line_no, line, src_path))
                continue

            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            if args.dry_run:
                continue

            if os.path.exists(dst_path) and os.path.getsize(dst_path) == os.path.getsize(src_path):
                continue

            shutil.copy2(src_path, dst_path)
            copied += 1

    missing_path = os.path.join(args.dst_root, "missing_depth_tiffs.txt")
    with open(missing_path, "w", encoding="utf-8") as f:
        for line_no, line, src_path in missing:
            f.write(f"line={line_no}\t{line}\t->\t{src_path}\n")

    dup_src_path = os.path.join(args.dst_root, "duplicates_same_depth_src_path.txt")
    with open(dup_src_path, "w", encoding="utf-8") as f:
        for src_path, refs in sorted(by_src_path.items()):
            if len(refs) <= 1:
                continue
            f.write(f"src_path: {src_path}\n")
            for (line_no, folder_token, frame_index, side_token) in refs:
                f.write(f"  line={line_no}\tfolder={folder_token}\tframe={frame_index}\tside={side_token}\n")
            f.write("\n")

    print("split_file:", split_file)
    print("src_root:", args.src_root)
    print("dst_root:", args.dst_root)
    print("dry_run:", args.dry_run)
    print("copied files:", copied)
    print("missing files:", len(missing))
    print("report missing:", missing_path)
    print("report duplicates (same src path):", dup_src_path)


if __name__ == "__main__":
    main()
