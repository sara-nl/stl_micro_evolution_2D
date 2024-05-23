method = "VTUNet"
# data
img_size = (12, 64, 64)
# model
patch_size = (4, 4, 4)
in_chans = 1
embed_dim = 96
depths = [2, 2, 2, 1]
num_heads = [3, 6, 12, 24]
window_size = (2, 7, 7)
zero_head = False
num_classes = 1  # output_channels
model_type = "gSTA" # check this
# training
lr = 1e-3
batch_size = 16
drop_path = 0
sched = "onecycle"
