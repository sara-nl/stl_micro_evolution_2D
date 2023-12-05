method = "VTUNet"

# kmc data
datafile = "kmc/exp_1_complete_2D.h5"
num_frames_per_experiment = 90
in_shape = [12, 1, 128, 128]
img_size = (12, 100, 100)
target_size = (128, 128)
pre_seq_length = 12
aft_seq_length = 12
total_length = 24

# model
patch_size = (4, 4, 4)
in_chans = 1
embed_dim = 96
depths = [2, 2, 2, 1]  # num layers
# depths_decoder = [1, 2, 2, 2]
num_heads = [3, 6, 12, 24]
window_size = (2, 7, 7)
zero_head = False
num_classes = 1  # output_channels
model_type = "gSTA"

# training
lr = 1e-3
batch_size = 16
drop_path = 0
sched = "onecycle"
