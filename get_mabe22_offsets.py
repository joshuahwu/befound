import os
import numpy as np


ABE_REINDEX = [6, 3, 0, 1, 2, 4, 5, 7, 8, 9, 10, 11]

N_KEYPTS = 12

SKELETON_CONFIG = {
    "KINEMATIC_TREE": [
        [0, 1, 2],          # center_back -> neck -> nose
        [1, 3],             # neck -> left_ear
        [1, 4],             # neck -> right_ear
        [1, 5],             # neck -> left_forepaw
        [1, 6],             # neck -> right_forepaw
        [0, 7],             # center_back -> left_hindpaw
        [0, 8],             # center_back -> right_hindpaw
        [0, 9, 10, 11],     # center_back -> tail_base -> tail_mid -> tail_tip
    ],
    "OFFSET": [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, -1, 0],
        [-1, 0, 0],
        [-1, 0, 0],
        [-1, 0, 0],
    ],
    "LABELS": [
        "center_back", "neck", "nose", "left_ear", "right_ear",
        "left_forepaw", "right_forepaw",
        "left_hindpaw", "right_hindpaw",
        "tail_base", "tail_middle", "tail_tip",
    ],
}


def load_mice_triplet(path):
    # load raw train data (with annotations for 2 tasks)
    data_train = np.load(
        os.path.join(path, "mouse_triplet_train.npy"), allow_pickle=True
    ).item()
    sequence_ids_train, sequence_data_train = zip(*data_train["sequences"].items())
    keypoints_train = np.stack([data["keypoints"] for data in sequence_data_train])

    # load submission data (no annotations)
    data_submission = np.load(
        os.path.join(path, "mouse_triplet_test.npy"), allow_pickle=True
    ).item()
    sequence_ids_submission, sequence_data_submission = zip(
        *data_submission["sequences"].items()
    )
    keypoints_submission = np.stack(
        [data["keypoints"] for data in sequence_data_submission]
    )

    # concatenate train and submission data
    sequence_ids = np.concatenate([sequence_ids_train, sequence_ids_submission], axis=0)
    keypoints = np.concatenate([keypoints_train, keypoints_submission], axis=0)

    split_mask = np.ones(len(sequence_ids), dtype=bool)
    split_mask[-len(sequence_ids_submission):] = False

    # treat each mouse independently, keep track of which video each mouse came from
    num_samples, sequence_length, num_mice, num_keypoints, _ = keypoints.shape
    keypoints = keypoints.transpose((0, 2, 1, 3, 4))
    keypoints = keypoints.reshape((-1, sequence_length, num_keypoints, 2))
    batch = np.repeat(np.arange(num_samples), num_mice)

    return keypoints, split_mask, batch

def get_parents(kinematic_tree, n_keypts):
    parents = [-1] * n_keypts
    for chain in kinematic_tree:
        for j in range(1, len(chain)):
            parents[chain[j]] = chain[j - 1]
    return parents


def compute_offsets_3d(
    keypoints,
    skeleton_config,
    missing_value: float = 0.0,
    method: str = "mean",
):
    pose = np.asarray(keypoints).reshape(-1, keypoints.shape[-2], keypoints.shape[-1])
    n_keypts = pose.shape[1]
    parents = get_parents(skeleton_config["KINEMATIC_TREE"], n_keypts)

    not_missing = np.any(pose != missing_value, axis=-1)

    offsets_3d = np.zeros(n_keypts, dtype=np.float32)
    n_valid = np.zeros(n_keypts, dtype=np.int64)
    for i in range(n_keypts):
        p = parents[i]
        if p < 0:
            continue

        valid = not_missing[:, i] & not_missing[:, p]
        n_valid[i] = int(valid.sum())
        if n_valid[i] == 0:
            raise ValueError(
                f"joint {i} ({skeleton_config['LABELS'][i]}) has no valid pred"
            )

        diff = pose[valid, i, :] - pose[valid, p, :]
        length = np.linalg.norm(diff, axis=-1)

        if method == "mean":
            offsets_3d[i] = float(length.mean())
        elif method == "median":
            offsets_3d[i] = float(np.median(length))
        else:
            raise ValueError(f"method 'mean' or 'median', but got {method!r}")

    return offsets_3d, n_valid


def print_offsets_3d(offsets_3d, skeleton_config, n_valid=None):
    parents = get_parents(skeleton_config["KINEMATIC_TREE"], len(offsets_3d))
    labels = skeleton_config["LABELS"]

    print("\nOFFSETS_3D = np.array(")
    print("    [")
    for i, v in enumerate(offsets_3d):
        if parents[i] < 0:
            comment = f"root ({labels[i]})"
        else:
            comment = f"{labels[parents[i]]} -> {labels[i]}"
        if n_valid is not None:
            comment += f"  [n={n_valid[i]}]"
        print(f"        {float(v):8.3f},  # {i:>2d}  {comment}")
    print("    ],")
    print("    dtype=np.float32,")
    print(")")
    print(f"\nOFFSETS_3D_SUM = {float(offsets_3d.sum()):.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute MABe22 OFFSETS_3D / OFFSETS_3D_SUM matching befound.data.constants."
    )
    parser.add_argument(
        "--data_path", required=True
    )
    parser.add_argument(
        "--method", default="mean", choices=["mean", "median"]
    )
    parser.add_argument(
        "--train_only", action="store_true"
    )
    args = parser.parse_args()

    keypoints, split_mask, batch = load_mice_triplet(args.data_path)
    keypoints = keypoints[:, :, ABE_REINDEX, :].astype(np.float32, copy=False)
    print(f"Loaded keypoints: shape={keypoints.shape}  (N*3, T, K=12, 2)")

    if args.train_only:
        per_mouse_train = split_mask[batch]      # (N*3,) True=train
        keypoints = keypoints[per_mouse_train]
        print(f"Restricting to train mice: shape={keypoints.shape}")

    offsets_3d, n_valid = compute_offsets_3d(
        keypoints, SKELETON_CONFIG, missing_value=0.0, method=args.method,
    )
    print_offsets_3d(offsets_3d, SKELETON_CONFIG, n_valid=n_valid)



# python get_mabe22_offsets.py --data_path /hpc/home/yw789/tdunn/befound_code/bams/data/mabe --method median --train_only