import os
import numpy as np


def read_skeleton(file_path):
    with open(file_path, 'r') as f:
        num_frames = int(f.readline())
        data = []

        for _ in range(num_frames):
            num_bodies = int(f.readline())

            bodies = []
            for _ in range(num_bodies):
                f.readline()  # body info (ignore)
                num_joints = int(f.readline())

                joints = []
                for _ in range(num_joints):
                    joint_info = list(map(float, f.readline().split()))
                    joints.append(joint_info[:3])  # (x, y, z)

                bodies.append(joints)

            # take first body only (common practice)
            if len(bodies) > 0:
                data.append(bodies[0])
            else:
                data.append([[0, 0, 0]] * 25)

    return np.array(data)  # shape: [T, 25, 3]

def main():
    input_dir = "/scratch/mgirishnair/Thesis/ntu_60-120_skeletons"
    output_dir = "/scratch/mgirishnair/Thesis/ntu_60-120_npy"
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".skeleton"):
            path = os.path.join(input_dir, file)

            skeleton = read_skeleton(path)

            save_path = os.path.join(output_dir, file.replace(".skeleton", ".npy"))
            np.save(save_path, skeleton)

if __name__== "__main__":
    main()
