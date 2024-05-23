method = "SimVP"
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = "gSTA"
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
# training
lr = 1e-3
batch_size = 16
drop_path = 0
sched = "onecycle"

yaml_file = "configs/kmc/experiments.yaml"
datafile = [
    ("exp_1_complete_2D.h5", 90, 4000),
    ("exp_2_complete_2D.h5", 55, 4000),
    ("exp_3_complete_2D.h5", 90, 4000),
]
seed = 0
random_split = True
sliding_win = 0
in_shape = [12, 1, 100, 100]
img_size = (12, 100, 100)
pre_seq_length = 12
aft_seq_length = 12
total_length = 24
