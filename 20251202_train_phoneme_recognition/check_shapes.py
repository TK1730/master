import numpy as np
import os


def check_shapes():
    with open('dataset/train.txt', 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f]

    print(f"Total files: {len(files)}")

    for i in range(min(10, len(files))):
        base_path = os.path.join('dataset', files[i])
        msp_path = base_path + "_msp.npy"

        if os.path.exists(msp_path):
            msp = np.load(msp_path)
            print(f"File {i}: {msp.shape}")
        else:
            print(f"File {i}: Not found {msp_path}")


if __name__ == "__main__":
    check_shapes()
