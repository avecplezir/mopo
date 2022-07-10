import sys

from dogo.score_model import score_model

DATA_PATHS = [
    "D3RLPY-MP4-P0-3.npy",
    "D3RLPY-MP4-P1-4.npy",
    "D3RLPY-PAP8-P0-3.npy",
    "D3RLPY-PAP8-P1-4.npy",
    # "MIXED-3.npy",
    # "D3RLPY-MP4-P0_20000.npy",
    # "D3RLPY-MP4-P1_20000.npy",
    # "D3RLPY-MP4-P2_20000.npy",
    # "D3RLPY-MP4-P3_20000.npy",
    # "D3RLPY-MP4-P4_20000.npy",
    # "D3RLPY-MP4-P0_100000.npy",
    # "D3RLPY-MP4-P1_100000.npy",
    # "D3RLPY-MP4-P2_100000.npy",
    # "D3RLPY-MP4-P3_100000.npy",
    # "D3RLPY-MP4-P4_100000.npy",
    # "D3RLPY-MP5-P0_20000.npy",
    # "D3RLPY-MP5-P1_20000.npy",
    # "D3RLPY-MP5-P2_20000.npy",
    # "D3RLPY-MP5-P3_20000.npy",
    # "D3RLPY-MP5-P4_20000.npy",
    # "D3RLPY-MP5-P0_100000.npy",
    # "D3RLPY-MP5-P1_100000.npy",
    # "D3RLPY-MP5-P2_100000.npy",
    # "D3RLPY-MP5-P3_100000.npy",
    # "D3RLPY-MP5-P4_100000.npy",
    # "D3RLPY-MP6-P0_20000.npy",
    # "D3RLPY-MP6-P1_20000.npy",
    # "D3RLPY-MP6-P2_20000.npy",
    # "D3RLPY-MP6-P3_20000.npy",
    # "D3RLPY-MP6-P4_20000.npy",
    # "D3RLPY-MP6-P0_100000.npy",
    # "D3RLPY-MP6-P1_100000.npy",
    # "D3RLPY-MP6-P2_100000.npy",
    # "D3RLPY-MP6-P3_100000.npy",
    # "D3RLPY-MP6-P4_100000.npy",
    # "D3RLPY-PAP8-P0_20000.npy",
    # "D3RLPY-PAP8-P1_20000.npy",
    # "D3RLPY-PAP8-P2_20000.npy",
    # "D3RLPY-PAP8-P3_20000.npy",
    # "D3RLPY-PAP8-P4_20000.npy",
    # "D3RLPY-PAP8-P0_100000.npy",
    # "D3RLPY-PAP8-P1_100000.npy",
    # "D3RLPY-PAP8-P2_100000.npy",
    # "D3RLPY-PAP8-P3_100000.npy",
    # "D3RLPY-PAP8-P4_100000.npy",
    # "D3RLPY-PAP9-P0_20000.npy",
    # "D3RLPY-PAP9-P1_20000.npy",
    # "D3RLPY-PAP9-P2_20000.npy",
    # "D3RLPY-PAP9-P3_20000.npy",
    # "D3RLPY-PAP9-P4_20000.npy",
    # "D3RLPY-PAP9-P0_100000.npy",
    # "D3RLPY-PAP9-P1_100000.npy",
    # "D3RLPY-PAP9-P2_100000.npy",
    # "D3RLPY-PAP9-P3_100000.npy",
    # "D3RLPY-PAP9-P4_100000.npy",
    # "D3RLPY-PAP10-P0_20000.npy",
    # "D3RLPY-PAP10-P1_20000.npy",
    # "D3RLPY-PAP10-P2_20000.npy",
    # "D3RLPY-PAP10-P3_20000.npy",
    # "D3RLPY-PAP10-P4_20000.npy",
    # "D3RLPY-PAP10-P0_100000.npy",
    # "D3RLPY-PAP10-P1_100000.npy",
    # "D3RLPY-PAP10-P2_100000.npy",
    # "D3RLPY-PAP10-P3_100000.npy",
    # "D3RLPY-PAP10-P4_100000.npy",
    # "RAND-1.npy",
    # "RAND-2.npy",
    # "RAND-3.npy",
    # "RAND-4.npy",
    # "RAND-5.npy",
    # "RAND-6.npy",
    # "RAND-7.npy",
    # "RAND-8.npy",
    # "RAND-9.npy",
    # "RAND-10.npy",
    # "RAND-D3RLPY-MP4-P0-1_100000.npy",
    # "RAND-D3RLPY-MP4-P1-1_100000.npy",
    # "RAND-D3RLPY-MP4-P2-1_100000.npy",
    # "RAND-D3RLPY-MP4-P3-1_100000.npy",
    # "RAND-D3RLPY-MP4-P4-1_100000.npy",
    # "RAND-D3RLPY-PAP8-P0-1_100000.npy",
    # "RAND-D3RLPY-PAP8-P1-1_100000.npy",
    # "RAND-D3RLPY-PAP8-P2-1_100000.npy",
    # "RAND-D3RLPY-PAP8-P3-1_100000.npy",
    # "RAND-D3RLPY-PAP8-P4-1_100000.npy",
    # "NOISE-D3RLPY-MP1-P0-1.npy",
    # "NOISE-D3RLPY-MP1-P1-1.npy",
    # "NOISE-D3RLPY-MP1-P2-1.npy",
    # "NOISE-D3RLPY-MP1-P3-1.npy",
    # "NOISE-D3RLPY-MP1-P4-1.npy",
    # "NOISE-D3RLPY-PAP5-P0-1.npy",
    # "NOISE-D3RLPY-PAP5-P1-1.npy",
    # "NOISE-D3RLPY-PAP5-P2-1.npy",
    # "NOISE-D3RLPY-PAP5-P3-1.npy",
    # "NOISE-D3RLPY-PAP5-P4-1.npy",
]

if __name__ == "__main__":
    experiment = sys.argv[1]
    score_model(experiment, DATA_PATHS)
