method = "VTUNet"

# kmc data
yaml_file = "configs/kmc/experiments.yaml"
datafile = [
    ("exp_1_complete_2D.h5", 90, 4000),
    ("exp_2_complete_2D.h5", 55, 4000),
    ("exp_3_complete_2D.h5", 90, 4000),
]
seed = 42
random_split = True
sliding_win = 0
in_shape = [12, 1, 128, 128]
img_size = (12, 100, 100)
pre_seq_length = 12
aft_seq_length = 12
total_length = 24

# model
patch_size = (4, 4, 4)  # best was (2, 4, 4)
in_chans = 1
embed_dim = 96
depths = [2, 2, 2, 1]
num_heads = [3, 6, 12, 24]
window_size = (7, 7, 7)  # was (2,7,7)
zero_head = False
num_classes = 200  # output_channels
model_type = "gSTA"

# training
batch_size = 16
drop_path = 0

lr = 1e-3
sched = "onecycle"
decay_epoch = 100
decay_rate = 0.1
warmup_lr = 1e-5
warmup_epoch = 0
final_div_factor = 10000.0

# logs
fp16 = True
