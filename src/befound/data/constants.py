import numpy as np

OFFSETS_3D = np.array(
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

OFFSETS_3D_SUM = np.sum(OFFSETS_3D)


OFFSETS_3D_MABE22 = np.array(
    [
           0.000,  #  0  root (center_back)  [n=0]
          25.632,  #  1  center_back -> neck  [n=7904572]
          25.495,  #  2  neck -> nose  [n=7904092]
          14.422,  #  3  neck -> left_ear  [n=7904197]
          14.318,  #  4  neck -> right_ear  [n=7902338]
          10.198,  #  5  neck -> left_forepaw  [n=7904532]
          10.198,  #  6  neck -> right_forepaw  [n=7904151]
          14.866,  #  7  center_back -> left_hindpaw  [n=7904473]
          15.000,  #  8  center_back -> right_hindpaw  [n=7904479]
          24.839,  #  9  center_back -> tail_base  [n=7903922]
          41.437,  # 10  tail_base -> tail_middle  [n=7893456]
          45.695,  # 11  tail_middle -> tail_tip  [n=7888304]
    ],
    dtype=np.float32,
)

OFFSETS_3D_MABE22_SUM = 242.100
OFFSETS_3D_MABE22_SUM_NO_TAIL = 242.100 - 41.437 - 45.695

MABE_REINDEX = [6, 3, 0, 1, 2, 4, 5, 7, 8, 9, 10, 11]