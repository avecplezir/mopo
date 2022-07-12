import sys

from dogo.score_model import score_model

DATA_PATHS = [
    "D3RLPY-MP1.npy",
    "D3RLPY-MP1_100000.npy",
    "D3RLPY-MP1-P0-3.npy",
    "D3RLPY-MP1-P1-4.npy",
    "D3RLPY-MP1-P0_25000.npy",
    "D3RLPY-MP1-P1_25000.npy",
    "D3RLPY-MP1-P2_25000.npy",
    "D3RLPY-MP1-P3_25000.npy",
    "D3RLPY-MP1-P4_25000.npy",
    "D3RLPY-MP1-P0_100000.npy",
    "D3RLPY-MP1-P1_100000.npy",
    "D3RLPY-MP1-P2_100000.npy",
    "D3RLPY-MP1-P3_100000.npy",
    "D3RLPY-MP1-P4_100000.npy",
    "D3RLPY-MP2-P0_100000.npy",
    "D3RLPY-MP2-P1_100000.npy",
    "D3RLPY-MP2-P2_100000.npy",
    "D3RLPY-MP2-P3_100000.npy",
    "D3RLPY-MP2-P4_100000.npy",
    "D3RLPY-MP3-P0_100000.npy",
    "D3RLPY-MP3-P1_100000.npy",
    "D3RLPY-MP3-P2_100000.npy",
    "D3RLPY-MP3-P3_100000.npy",
    "D3RLPY-MP3-P4_100000.npy",
    "D3RLPY-PAP5.npy",
    "D3RLPY-PAP5_100000.npy",
    "D3RLPY-PAP5-P0-3.npy",
    "D3RLPY-PAP5-P1-4.npy",
    "D3RLPY-PAP5-P0_25000.npy",
    "D3RLPY-PAP5-P1_25000.npy",
    "D3RLPY-PAP5-P2_25000.npy",
    "D3RLPY-PAP5-P3_25000.npy",
    "D3RLPY-PAP5-P4_25000.npy",
    "D3RLPY-PAP5-P0_100000.npy",
    "D3RLPY-PAP5-P1_100000.npy",
    "D3RLPY-PAP5-P2_100000.npy",
    "D3RLPY-PAP5-P3_100000.npy",
    "D3RLPY-PAP5-P4_100000.npy",
    "D3RLPY-PAP6-P0_100000.npy",
    "D3RLPY-PAP6-P1_100000.npy",
    "D3RLPY-PAP6-P2_100000.npy",
    "D3RLPY-PAP6-P3_100000.npy",
    "D3RLPY-PAP6-P4_100000.npy",
    "D3RLPY-PAP7-P0_100000.npy",
    "D3RLPY-PAP7-P1_100000.npy",
    "D3RLPY-PAP7-P2_100000.npy",
    "D3RLPY-PAP7-P3_100000.npy",
    "D3RLPY-PAP7-P4_100000.npy",
    "RAND-1.npy",
    "RAND-2.npy",
    "RAND-3.npy",
    "RAND-4.npy",
    "RAND-5.npy",
    "RAND-6.npy",
    "RAND-7.npy",
    "RAND-8.npy",
    "RAND-9.npy",
    "RAND-10.npy",
    "RAND-D3RLPY-MP1-P0-1.npy",
    "RAND-D3RLPY-MP1-P1-1.npy",
    "RAND-D3RLPY-MP1-P2-1.npy",
    "RAND-D3RLPY-MP1-P3-1.npy",
    "RAND-D3RLPY-MP1-P4-1.npy",
    "RAND-D3RLPY-PAP5-P0-1.npy",
    "RAND-D3RLPY-PAP5-P1-1.npy",
    "RAND-D3RLPY-PAP5-P2-1.npy",
    "RAND-D3RLPY-PAP5-P3-1.npy",
    "RAND-D3RLPY-PAP5-P4-1.npy",
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