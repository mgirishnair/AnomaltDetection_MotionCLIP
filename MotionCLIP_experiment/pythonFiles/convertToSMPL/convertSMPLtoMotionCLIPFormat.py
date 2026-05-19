import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed

from NTU_actions import NTU_ACTIONS

NUM_WORKERS = 8
PRINT_EVERY = 5000 #for now 20, change it to 500 or something
CHUNK_SIZE = 1000

INPUT_DIR = r"/scratch/mgirishnair/Pose_to_SMPL/fit/output/NTU_120"
OUTPUT_DIR = r"/scratch/mgirishnair/Thesis/MotionCLIP_ready_datasetFinalAll_120"
PKL_LIST_FILE = r"/scratch/mgirishnair/Thesis/smpl_pkl_files_120.txt"
TARGET_FRAMES = 60

def process_file_safe(pkl_path: str):
    try:
        # if os.path.getsize(pkl_path) < 1000:
        #     return None, f"[SKIP] File too small: {pkl_path}"

        motion, label, label_name = process_file(pkl_path)
        name = os.path.splitext(os.path.basename(pkl_path))[0]
        return (motion, label, label_name, name), None

    except (EOFError, pickle.UnpicklingError, OSError, KeyError, ValueError) as e:
        return None, f"[SKIP] Failed to process {pkl_path}: {e}"

def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta = torch.norm(axis_angle, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = axis_angle / theta

    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)

    K = torch.stack([
        zeros, -z,    y,
        z,     zeros, -x,
        -y,    x,     zeros
    ], dim=-1).reshape(axis.shape[:-1] + (3, 3))

    I = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    I = I.expand(axis.shape[:-1] + (3, 3))

    theta = theta[..., 0][..., None, None]
    R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    return R


def matrix_to_rot6d(rotmat: torch.Tensor) -> torch.Tensor:
    return rotmat[..., :, :2].reshape(*rotmat.shape[:-2], 6)


# def resample_to_60(seq: torch.Tensor, target_frames: int = 60) -> torch.Tensor:
#     # seq: [T, 24, 6]
#     T, J, C = seq.shape
#
#     if T == target_frames:
#         return seq
#
#     x = seq.permute(1, 2, 0).reshape(1, J * C, T)  # [1, 144, T]
#
#     if T == 1:
#         x = x.repeat(1, 1, target_frames)
#     else:
#         x = F.interpolate(x, size=target_frames, mode="linear", align_corners=True)
#
#     x = x.reshape(J, C, target_frames).permute(2, 0, 1)  # [60, 24, 6]
#     return x
def resample_seq(seq: torch.Tensor, target_frames: int = 60) -> torch.Tensor:
    # seq: [T, ...]
    T = seq.shape[0]

    if T == target_frames:
        return seq

    x = seq.reshape(T, -1).transpose(0, 1).unsqueeze(0)  # [1, C, T]

    if T == 1:
        x = x.repeat(1, 1, target_frames)
    else:
        x = F.interpolate(x, size=target_frames, mode="linear", align_corners=True)

    x = x.squeeze(0).transpose(0, 1).reshape(target_frames, *seq.shape[1:])
    return x

def process_file(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    pose = np.asarray(data["pose_params"], dtype=np.float32).reshape(-1, 24, 3)  # [T, 24, 3]
    Jtr = np.asarray(data["Jtr"], dtype=np.float32)                               # [T, 24, 3]
    trans = Jtr[:, 0, :]                                                          # [T, 3]

    label_name = str(data["label"]).strip()
    if label_name not in NTU_ACTIONS:
        raise KeyError(f"Label '{label_name}' not found in NTU_ACTIONS")

    label = NTU_ACTIONS[label_name]

    pose = torch.tensor(pose, dtype=torch.float32)     # [T, 24, 3]
    trans = torch.tensor(trans, dtype=torch.float32)   # [T, 3]

    # Resample first
    pose = resample_seq(pose, TARGET_FRAMES)           # [60, 24, 3]
    trans = resample_seq(trans, TARGET_FRAMES)         # [60, 3]

    # Then convert pose to rot6d
    rotmat = axis_angle_to_matrix(pose)                # [60, 24, 3, 3]
    rot6d = matrix_to_rot6d(rotmat)                    # [60, 24, 6]

    # Append translation as 25th slot
    trans6d = torch.zeros((TARGET_FRAMES, 1, 6), dtype=torch.float32)
    trans6d[:, 0, :3] = trans                          # [60, 1, 6]

    motion = torch.cat([rot6d, trans6d], dim=1)       # [60, 25, 6]

    return motion.numpy().astype(np.float32), label, label_name

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Read list of pkl files
    with open(PKL_LIST_FILE, "r", encoding="utf-8") as f:
        listed_files = [line.strip() for line in f if line.strip()]

    pkl_files = []
    for name in listed_files:
        pkl_path = name if os.path.isabs(name) else os.path.join(INPUT_DIR, name)
        if os.path.isfile(pkl_path):
            pkl_files.append(pkl_path)
        else:
            print(f"[SKIP] Missing file: {pkl_path}")

    if not pkl_files:
        raise FileNotFoundError(f"No valid .pkl files found from list: {PKL_LIST_FILE}")

    print(f"Found {len(pkl_files)} PKL files")

    valid_files = []
    for pkl_path in pkl_files:
        if os.path.getsize(pkl_path) < 1000:
            print(f"[SKIP] File too small: {pkl_path}")
            continue
        valid_files.append(pkl_path)

    if not valid_files:
        raise RuntimeError("No valid files to process")

    n = len(valid_files)

    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="motionclip_chunks_", dir=OUTPUT_DIR)

    chunk_idx = 0
    chunk_data = []
    chunk_labels = []
    chunk_names = []
    chunk_label_names = []

    written = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        future_to_path = {ex.submit(process_file_safe, p): p for p in valid_files}

        for i, fut in enumerate(as_completed(future_to_path), start=1):
            result, err = fut.result()

            if err is not None:
                print(err)
                skipped += 1
            else:
                motion, label, label_name, name = result

                chunk_data.append(motion)
                chunk_labels.append(label)
                chunk_names.append(name)
                chunk_label_names.append(label_name)

                written += 1

            if len(chunk_data) >= CHUNK_SIZE:

                np.save(
                    os.path.join(tmp_dir, f"X_chunk_{chunk_idx:05d}.npy"),
                    np.stack(chunk_data, axis=0).astype(np.float32),
                )

                np.save(
                    os.path.join(tmp_dir, f"y_chunk_{chunk_idx:05d}.npy"),
                    np.array(chunk_labels, dtype=np.int64),
                )

                with open(
                    os.path.join(tmp_dir, f"names_chunk_{chunk_idx:05d}.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for x in chunk_names:
                        f.write(x + "\n")

                with open(
                    os.path.join(tmp_dir, f"label_names_chunk_{chunk_idx:05d}.txt"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    for x in chunk_label_names:
                        f.write(x + "\n")

                chunk_idx += 1
                chunk_data.clear()
                chunk_labels.clear()
                chunk_names.clear()
                chunk_label_names.clear()

            if i % PRINT_EVERY == 0 or i == n:
                print(f"[{i}/{n}] written={written}, skipped={skipped}")

    # flush remaining chunk
    if chunk_data:
        np.save(
            os.path.join(tmp_dir, f"X_chunk_{chunk_idx:05d}.npy"),
            np.stack(chunk_data, axis=0).astype(np.float32),
        )

        np.save(
            os.path.join(tmp_dir, f"y_chunk_{chunk_idx:05d}.npy"),
            np.array(chunk_labels, dtype=np.int64),
        )

        with open(
            os.path.join(tmp_dir, f"names_chunk_{chunk_idx:05d}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for x in chunk_names:
                f.write(x + "\n")

        with open(
            os.path.join(tmp_dir, f"label_names_chunk_{chunk_idx:05d}.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            for x in chunk_label_names:
                f.write(x + "\n")

    X_chunk_files = sorted(
        f for f in os.listdir(tmp_dir) if f.startswith("X_chunk_")
    )
    y_chunk_files = sorted(
        f for f in os.listdir(tmp_dir) if f.startswith("y_chunk_")
    )

    if not X_chunk_files:
        raise RuntimeError("No valid files processed")

    total_written = 0
    for fname in X_chunk_files:
        arr = np.load(os.path.join(tmp_dir, fname), mmap_mode="r")
        total_written += arr.shape[0]

    X_path = os.path.join(OUTPUT_DIR, "X.npy")
    y_path = os.path.join(OUTPUT_DIR, "y.npy")

    X = np.lib.format.open_memmap(
        X_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_written, TARGET_FRAMES, 25, 6),
    )

    y = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.int64,
        shape=(total_written,),
    )

    names = []
    label_names = []

    write_idx = 0

    for x_file, y_file in zip(X_chunk_files, y_chunk_files):

        x_arr = np.load(os.path.join(tmp_dir, x_file), mmap_mode="r")
        y_arr = np.load(os.path.join(tmp_dir, y_file), mmap_mode="r")

        m = x_arr.shape[0]

        X[write_idx:write_idx + m] = x_arr
        y[write_idx:write_idx + m] = y_arr

        write_idx += m

        idx = x_file.replace("X_chunk_", "").replace(".npy", "")

        with open(os.path.join(tmp_dir, f"names_chunk_{idx}.txt")) as f:
            names.extend([line.strip() for line in f])

        with open(os.path.join(tmp_dir, f"label_names_chunk_{idx}.txt")) as f:
            label_names.extend([line.strip() for line in f])

    X.flush()
    y.flush()

    with open(os.path.join(OUTPUT_DIR, "filenames.txt"), "w") as f:
        for name in names:
            f.write(name + "\n")

    with open(os.path.join(OUTPUT_DIR, "label_names.txt"), "w") as f:
        for label_name in label_names:
            f.write(label_name + "\n")

    shutil.rmtree(tmp_dir)

    print("Final X shape:", (total_written, TARGET_FRAMES, 25, 6))
    print("Final y shape:", (total_written,))

if __name__ == "__main__":
    main()
