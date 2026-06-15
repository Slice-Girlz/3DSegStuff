from ThreeDSegStuff.train import train
from ThreeDSegStuff.config_unet import model
from ThreeDSegStuff.loss import loss

loss  = ...
optimizer = ...

train(
    # model, 
    # loss, 
    # optimizer,
    input_dir = '/mnt/efs/dl_jrc/student_data/S-MS/annotations_omezarr/',
    output_dir = '.',
    n_training_steps = 10,
    channel = 1,
    input_shape = [1, 16, 128, 128],
    output_shape = [1, 14, 124, 124],
    batch_size = 1, 
    prob_augment = 0.3, 
    var_noise = 10e-5,
    neighborhood = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    save_snapshots_every = 1)