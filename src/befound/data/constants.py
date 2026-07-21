import numpy as np

OFFSETS_3D_BOX = np.array(
                        [
                            0.0,
                            17.0,
                            14.0,
                            19.0,
                            24.0,
                            42.0,
                            23.5,
                            11.5,
                            3.0,
                            23.5,
                            11.5,
                            3.0,
                            29.5,
                            17.0,
                            11.0,
                            29.5,
                            17.0,
                            11.0,
                        ],
                        dtype=np.float32,
                    )

OFFSETS_3D_SUM_BOX = np.sum(OFFSETS_3D_BOX)


OFFSETS_3D_PAIRR24M = np.array(
    [
        0.0,
        40.933956,
        60.553497,
        34.144684,
        21.212831,
        33.003525,
        35.15831,
        40.9036,
        31.277847,
        31.277847,
        24.26066,
        24.26066,
    ],
    dtype=np.float32,
)

OFFSETS_3D_SUM_PAIRR24M = np.sum(OFFSETS_3D_PAIRR24M)