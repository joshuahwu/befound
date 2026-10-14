import math
from typing import Optional

import numpy as np

OFFSETS_3D = {
    # Updated from the segment lengths befound/scripts/compute_seg_lengths.py
    # computed over each dataset's full train split (see
    # befound/scripts/bash/slurm/7180329.out). wu_iclr25 and pairr24m's
    # previous entries below were rough/placeholder estimates (notably
    # wu_iclr25's suspiciously round numbers) that differed from the
    # computed values by up to ~51% and ~19% respectively on individual
    # segments; virtual_rodent's previous entry already matched closely
    # (<1%), consistent with having been computed the same way before.
    #
    # For every dataset, each genuinely mirrored L/R segment pair (same
    # parent-relationship type on both sides, e.g. ShoulderL<-SpineF and
    # ShoulderR<-SpineF, or ElbowL<-ShoulderL and ElbowR<-ShoulderR) has its
    # two independently-computed lengths averaged and shared by both sides.
    # EarL/EarR is deliberately NOT averaged this way: per each skeleton's
    # own KINEMATIC_TREE, EarR's parent is EarL (not e.g. SpineF/NeckTop),
    # so its "segment length" measures the ear-to-ear distance, not a second
    # copy of the same NeckTop/SpineF-to-ear bone as EarL -- the two aren't
    # mirrors of each other at all, so they keep their own independent,
    # unaveraged computed values.
    "wu_iclr25": np.array(
        [
            0.0,
            18.488229751586914,
            14.586752891540527,
            19.19923973083496,
            24.3575496673584,
            42.75708770751953,
            23.76425552368164,
            11.926959991455078,
            4.381002426147461,
            23.76425552368164,
            11.926959991455078,
            4.381002426147461,
            29.622291564941406,
            17.792160034179688,
            11.583467483520508,
            29.622291564941406,
            17.792160034179688,
            11.583467483520508,
        ],
        dtype=np.float32,
    ),
    "pairr24m": np.array(
        [
            0.0,
            42.23212432861328,
            63.319271087646484,
            35.403526306152344,
            25.15140724182129,
            35.30955123901367,
            36.6869010925293,
            42.0401496887207,
            34.862510681152344,
            34.862510681152344,
            25.064403533935547,
            25.064403533935547,
        ],
        dtype=np.float32,
    ),
    # Legacy 44-keypoint skeleton -- distinct from, and not comparable to,
    # the current 29-keypoint "mouse_ephys" below (different keypoint
    # count/skeleton entirely). Left untouched.
    "mouse44_ephys": np.array(
        [
            0.0,
            16.13878,
            7.5940557,
            24.745075,
            6.8422766,
            7.6958966,
            8.700052,
            13.791159,
            8.046115,
            9.305047,
            13.620974,
            18.468988,
            17.429226,
            24.179575,
            51.73884,
            2.322974,
            17.325264,
            8.419868,
            3.1635725,
            9.633051,
            4.125996,
            2.5192335,
            3.6711023,
            17.498499,
            8.489913,
            3.5101671,
            9.958358,
            4.1205425,
            2.5536642,
            3.7212732,
            7.658827,
            17.67529,
            3.9156158,
            12.569995,
            9.95836,
            3.920858,
            6.7193823,
            7.800202,
            17.69783,
            4.3090596,
            12.971825,
            9.670808,
            3.9902375,
            6.5995197,
        ],
        dtype=np.float32,
    ),
    "virtual_rodent": np.array(
        [
            0.0,
            42.44869613647461,
            47.47469711303711,
            39.809444427490234,
            28.40323257446289,
            45.753902435302734,
            38.89545822143555,
            22.93250274658203,
            25.64581298828125,
            18.487703323364258,
            12.562999725341797,
            22.93250274658203,
            25.64581298828125,
            18.487703323364258,
            12.562999725341797,
            22.474756240844727,
            28.031417846679688,
            32.20182800292969,
            21.897361755371094,
            22.474756240844727,
            28.031417846679688,
            32.20182800292969,
            21.897361755371094,
        ],
        dtype=np.float32,
    ),
    "sdannce": np.array(
        [
            0.0,
            55.5916862487793,
            31.599958419799805,
            28.31549072265625,
            48.7895393371582,
            50.74927520751953,
            50.94810485839844,
            22.694507598876953,
            27.73590850830078,
            19.436065673828125,
            16.041728973388672,
            22.694507598876953,
            27.73590850830078,
            19.436065673828125,
            16.041728973388672,
            29.12398910522461,
            36.732147216796875,
            26.27729034423828,
            26.75732421875,
            29.12398910522461,
            36.732147216796875,
            26.27729034423828,
            26.75732421875,
        ],
        dtype=np.float32,
    ),
    "mouse_ephys": np.array(
        [
            0.0,
            24.855926513671875,
            8.845537185668945,
            10.56612777709961,
            21.13225555419922,
            24.860774993896484,
            18.636341094970703,
            17.20271110534668,
            23.91180992126465,
            10.978093147277832,
            8.714558601379395,
            9.838094711303711,
            4.040094375610352,
            2.4900684356689453,
            10.978093147277832,
            8.714558601379395,
            9.838094711303711,
            4.040094375610352,
            2.4900684356689453,
            7.166618824005127,
            17.463489532470703,
            12.96878433227539,
            10.0386323928833,
            3.9038755893707275,
            7.166618824005127,
            17.463489532470703,
            12.96878433227539,
            10.0386323928833,
            3.9038755893707275,
        ],
        dtype=np.float32,
    ),
}

# OFFSETS_3D["virtual_rodent"] = np.zeros(
#     23, dtype=np.float32
# )  # Placeholder; set after examining data

# OFFSETS_3D_SUM = {
#     dataset_name: np.sum(offsets) for dataset_name, offsets in OFFSETS_3D.items()
# }

DATA_FPS = {
    "pairr24m": 30,
    "wu_iclr25": 90,
    "mouse_ephys": 100,
    "virtual_rodent": 50,
    "sdannce": 50,
    "mabe22": 30,
    "calms21": 30,
}


SEQUENCE_SCALE = 3

BODY_SCALE = {
    "sdannce":232.724308,
    # "sdannce_CNTNAP":453.930418,
    "pairr24m":167.816267,
    "virtual_rodent_base":[193.576889, 115], # second number is number of sessions
    "virtual_rodent_xsession":[195.520340,34],
    "virtual_rodent_xsubject":[196.835215,60],
    "virtual_rodent":194.828453,
    "mouse_ephys":85.760954,
    "wu_iclr25":88.277407,
}

# Multiplier used by befound.train.losses.get_batch_loss to rescale jpe/root
# back up when body-scale normalization is active: RodentDiskDataset
# divides pose by body_scale (~O(100-300)) before those losses ever see it,
# shrinking their squared error by that same factor and making their
# relative weight against e.g. "prior" much smaller than in previous,
# unnormalized training runs. A few of BODY_SCALE's entries are
# [mean, n_sessions] pairs for individual virtual_rodent x_splits, not real
# per-dataset body scales (see RodentDiskDataset.__init__) -- excluded
# here, only the five actual per-dataset values are averaged.
BODY_SCALE_LOSS_FACTOR = SEQUENCE_SCALE * np.mean(
    [v for v in BODY_SCALE.values() if not isinstance(v, (list, tuple))]
)

# The downstream dataloader always feeds the model model_window frames (the
# model's own "window" config, e.g. model.window in model_config.yaml), but
# each sample optionally gets interpolated to a random framerate between
# min_interp_fps and its dataset's native fps before that (see
# befound.data.train_utils.interpolate_center_window; min_interp_fps comes
# from train.augmentations.fps.min_fps when that augmentation is configured).
# model_window frames at the slowest end (min_interp_fps) span
# model_window/min_interp_fps seconds, so each dataset's on-disk window needs
# to cover *at least* that much real time -- at its own native fps -- for
# min_interp_fps interpolation to have enough source frames to draw from.


def get_window_size(
    dataset_name: str, model_window: int, min_interp_fps: Optional[int] = None
) -> int:
    """Native-fps frame count needed to cover model_window frames at
    min_interp_fps, forced odd so a window has a single, well-defined center
    frame. If min_interp_fps is None (no fps augmentation configured), this
    is just model_window itself.

    model_window and min_interp_fps should always be read from the same
    model config (model.window, train.augmentations.fps.min_fps) used
    everywhere else in the pipeline -- both the train/val/test split
    generator (generate_train_val_test_splits.py) and RodentDiskDataset call
    this with those same config-derived values, so window size is always
    *derived* from the model config, never hardcoded or passed between the
    two independently.
    """
    if min_interp_fps is None:
        return model_window
    window = math.ceil(model_window / min_interp_fps * DATA_FPS[dataset_name])
    if window % 2 == 0:
        window += 1  # odd-length windows are easier to center on a frame
    return window

DATA_WINDOW = {
    dataset_name: get_window_size(dataset_name, model_window=51, min_interp_fps=25)
    for dataset_name in DATA_FPS.keys()
}

DEFAULT_AUGMENTATIONS = {
    "2d_project": {"prob": 0.2},
    "fps": {"max_fps": 100, "min_fps": 25, "mode": "cubic"},
    "kpt_ablation": {"max_ablate": 5, "prob": 0.1},
    "kpt_shuffle": True,
    "rotation": {"pole_concentration": 2},
    "temporal_mask_future": {"prob": 0.1},
    "temporal_mask_interior": {"max_ablate": 10, "prob": 0.1},
    "temporal_mask_past": {"prob": 0.1},
}