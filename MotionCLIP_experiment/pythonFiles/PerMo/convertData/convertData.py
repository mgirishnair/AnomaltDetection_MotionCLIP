from pathlib import Path
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def resample_indices(num_frames: int, target_frames: int = 60) -> np.ndarray:
    return np.linspace(0, num_frames - 1, target_frames).round().astype(int)


def axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    """
    axis_angle: [T, J, 3]
    returns:    [T, J, 6]
    """
    T, J, _ = axis_angle.shape

    rot_mats = R.from_rotvec(axis_angle.reshape(-1, 3)).as_matrix()
    rot_mats = rot_mats.reshape(T, J, 3, 3)

    # first two columns of each rotation matrix
    rot6d = rot_mats[..., :, :2].reshape(T, J, 6)

    return rot6d.astype(np.float32)


def convert_one_npz(input_path: Path, output_path: Path, target_frames: int = 60):
    data = np.load(input_path, allow_pickle=True)

    if "poses" not in data:
        raise KeyError(f"'poses' not found in {input_path}")

    poses = data["poses"]  # [T, 156] = [T, 52 * 3]
    T = poses.shape[0]

    if poses.shape[1] != 156:
        raise ValueError(f"Expected poses shape [T, 156], got {poses.shape} in {input_path}")

    poses = poses.reshape(T, 52, 3)

    # SMPL-H body only: root + 21 body joints
    poses_body = poses[:, :22, :]  # [T, 22, 3]

    # Pad 3 identity-rotation joints to get MotionCLIP-style [T, 25, 3]
    pad = np.zeros((T, 3, 3), dtype=poses_body.dtype)
    poses_25 = np.concatenate([poses_body, pad], axis=1)

    # Resample to 60 frames
    idx = resample_indices(T, target_frames)
    poses_25 = poses_25[idx]

    trans = data["trans"][idx] if "trans" in data else np.zeros((target_frames, 3), dtype=np.float32)

    # Axis-angle -> rot6D
    motion = axis_angle_to_rot6d(poses_25)  # [60, 25, 6]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        motion=motion.astype(np.float32),
        trans=trans.astype(np.float32),
        betas=data["betas"].astype(np.float32) if "betas" in data else None,
        gender=data["gender"] if "gender" in data else None,
        mocap_framerate=data["mocap_framerate"] if "mocap_framerate" in data else None,
        source_file=str(input_path),
    )


def convert_dataset(input_root: Path, output_root: Path, target_frames: int = 60):
    npz_files = sorted(input_root.rglob("*.npz"))

    print(f"Found {len(npz_files)} .npz files")

    failed = []

    for i, input_path in enumerate(npz_files, start=1):
        rel_path = input_path.relative_to(input_root)
        output_path = output_root / rel_path

        try:
            convert_one_npz(input_path, output_path, target_frames)
            print(f"[{i}/{len(npz_files)}] Converted: {rel_path}")
        except Exception as e:
            print(f"[{i}/{len(npz_files)}] FAILED: {rel_path}")
            print(f"  Error: {e}")
            failed.append((str(rel_path), str(e)))

    print("\nDone.")
    print(f"Converted: {len(npz_files) - len(failed)}")
    print(f"Failed:    {len(failed)}")

    if failed:
        fail_log = output_root / "failed_files.txt"
        with open(fail_log, "w", encoding="utf-8") as f:
            for path, error in failed:
                f.write(f"{path}\t{error}\n")
        print(f"Failure log saved to: {fail_log}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="Condition")
    parser.add_argument("--output_root", type=str, default="Condition_MotionCLIP")
    parser.add_argument("--target_frames", type=int, default=60)

    args = parser.parse_args()

    convert_dataset(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        target_frames=args.target_frames,
    )
