"""
Compute mean segment (bone) lengths per dataset from each dataset's train
split, based on ceph/data/sdannce/code/get_seg_lengths.py.

Rather than loading each dataset's full pose array (100M+ frames for
sdannce/pairr24m) or materializing every frame referenced by train_inds at
once (also 10s of millions of frames -- np.unique(train_inds) turns out to
cover close to half of sdannce's raw frames, not a small subsample), this
uses befound.data.data.RodentDiskDataset purely to open/locate the right h5
file(s) per dataset (it already knows how to assemble sdannce/pairr24m's
a1/a2/single multi-file structure) and streams every *unique* frame index
referenced in its loaded train indices through in batches, computing segment
lengths on the GPU per batch and accumulating a running sum/count for the
mean -- so memory use stays bounded by BATCH_SIZE regardless of dataset size.

mouse_ephys is skipped -- its train/val/test split doesn't exist yet.
"""

import numpy as np
import torch
from neuroposelib.io import read

from befound.data import get_segment_len_torch
from befound.data.data import RodentDiskDataset

DATA_PATH = "/mnt/home/jwu10/working/ceph/data/"
DATASETS = ["sdannce", "pairr24m", "wu_iclr25", "virtual_rodent", "mouse_ephys"]
BATCH_SIZE = 65536
DEVICE = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"

def _accumulate_segment_lengths(pose_h5, indices, kinematic_tree, offset, n_kpts):
    """Stream indices through pose_h5 (an open h5py Dataset) in
    BATCH_SIZE chunks, computing segment lengths on DEVICE per batch. Returns
    (sum_of_segment_lengths, n_frames_processed) for the caller to accumulate
    into its own running totals."""
    indices = np.sort(np.asarray(indices))
    sum_seg_len = torch.zeros(n_kpts, device=DEVICE)
    n_processed = 0
    for start in range(0, len(indices), BATCH_SIZE):
        idx_batch = indices[start : start + BATCH_SIZE]  # sorted -> valid h5py fancy index
        pose_np = np.asarray(pose_h5[idx_batch], dtype=np.float32)  # (b, n_kpts, 3)
        pose = torch.from_numpy(pose_np).to(DEVICE)

        offsets = get_segment_len_torch(pose, kinematic_tree, offset)  # (b, n_kpts, 3)
        seg_len = offsets.abs().sum(-1)  # (b, n_kpts), matches get_seg_lengths.py's metric

        sum_seg_len += seg_len.sum(0)
        n_processed += seg_len.shape[0]

    return sum_seg_len, n_processed


def main():
    print(f"device={DEVICE}")
    for name in DATASETS:
        base = f"{DATA_PATH}{name}/"
        skeleton_config = read.config(base + "skeleton.yaml")
        kinematic_tree = skeleton_config["KINEMATIC_TREE"]
        offset = torch.tensor(skeleton_config["OFFSET"], dtype=torch.float32, device=DEVICE)
        n_kpts = offset.shape[0]

        # stride=1 / no speed-outlier filtering: walk every window in the
        # dataset's train split, matching the "use np.unique() on train_inds"
        # intent as closely as the data will allow.
        dataset = RodentDiskDataset(
            data_path=DATA_PATH,
            dataset_names=[name],
            train_val_test="train",
            strides={name: 1},
            use_pe_indices=True,
            model_window=51,
            min_interp_fps=25,
            balanced_sampling=False,
            use_default_seg_len=False,
            use_speed_filtered_inds=False,
            scale_body=False,
        )

        running_sum = torch.zeros(n_kpts, device=DEVICE)
        running_count = 0

        for h5_key in dataset.h5_paths.keys():
            print(f"Processing {h5_key} for {name} dataset...")
            pose_h5 = dataset.h5_datasets[h5_key]
            inds = dataset.indices[h5_key]
            print(
                f"[{name}] {h5_key}: {len(inds)} unique frames "
                f"(of {pose_h5.shape[0]} total in file)"
            )

            sum_seg_len, n_processed = _accumulate_segment_lengths(
                pose_h5, inds, kinematic_tree, offset, n_kpts
            )
            running_sum += sum_seg_len
            running_count += n_processed

        mean_seg_len = (running_sum / running_count).cpu().numpy()
        print(f"[{name}] n_frames_used={running_count}")
        print(f"[{name}] segment_lengths={mean_seg_len.tolist()}")
        print()

        dataset.close()


if __name__ == "__main__":
    main()
