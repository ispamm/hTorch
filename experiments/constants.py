[dataset]
batch_size = 2
repetitions = 200
shuffle = false
train_split = 0.6
test_split = 0.5
width = 384
height = 384
data_size_train = 1000
data_size_val = 100

[psp]
deep_supervision = false
dropout = 0.1
classes = 10
layers = 101
alpha_aux = 0

[training]
learning_rate = 3e-4
num_epochs = 20

[loss]
alpha = 0.7
beta = 0.3
gamma = 0.75

[crf]
max_iter = 2
pos_w = 3
pos_xy_std = 3
bi_xy_std = 3
bi_rgb_std = 3
bi_w = 3

