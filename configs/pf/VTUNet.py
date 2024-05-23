method = "VTUNet"
# data
img_size = (10, 64, 64)
pre_seq_length = 12
aft_seq_length = 12
# model
patch_size = (2, 4, 4)
in_chans = 1
embed_dim = 96
depths = [2, 2, 2, 1]
num_heads = [3, 6, 12, 24]
window_size = (2, 7, 7)
zero_head = False
num_classes = 1  # output_channels
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
