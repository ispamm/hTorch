import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import configparser
import os
import glob
import sys
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
from torchmetrics import IoU, F1
from torchgeometry.losses import TverskyLoss
from sklearn.metrics import jaccard_similarity_score

sys.path.append('hTorch/')
sys.path.append('pytorch-image-models/')

from madgrad import MADGRAD
from kaggle_funcs import stick_all_train, get_patches, predict_id, calc_jacc
from loss import FocalTverskyLoss

parser = argparse.ArgumentParser(description='htorch training and testing')
parser.add_argument('-m', '--model', help='model to train, choose from: {psp, swin, unet}', required=True)
parser.add_argument('-q', '--quaternion', help='wheter to use quaternions', action='store_true', default=False)
parser.add_argument('-s', '--save-dir', help='where to save checkpoint files', required=True)
parser.add_argument('-w', '--checkpoint-weight-path', help='saved checkpoint weight file to resume trainng from')
parser.add_argument('-o', '--checkpoint-optim-path', help='saved checkpoint optimizer to resume trainng from')
parser.add_argument('-t', '--test', help='test mode', action='store_true', default=False)
parser.add_argument('-l', '--save-last', help='save only last epoch', action='store_true', default=False)

args = parser.parse_args()

def to_rgb(input):
    img = np.zeros(tuple([input.shape[1]]) + tuple([input.shape[2]]) + tuple([3]))
    img[:, :, 0] = input[4]
    img[:, :, 1] = input[2]
    img[:, :, 2] = input[1]

    return (img * 255).astype(np.uint8)

def plot_fig(input, name):
    
    plt.figure(figsize=[20, 20])
    fig, axes = plt.subplots(2,5)
    axes[0][0].imshow(input[0], cmap="gray")
    axes[0][0].title.set_text("Buildings")

    axes[0][1].imshow(input[1], cmap="gray")
    axes[0][1].title.set_text("Man-made struct")

    axes[0][2].imshow(input[2], cmap="gray")
    axes[0][2].title.set_text("Road")

    axes[0][3].imshow(input[3], cmap="gray")
    axes[0][3].title.set_text("Track")

    axes[0][4].imshow(input[4], cmap="gray")
    axes[0][4].title.set_text("Trees")
    
    axes[1][0].imshow(input[5], cmap="gray")
    axes[1][0].title.set_text("Crops")

    axes[1][1].imshow(input[6], cmap="gray")
    axes[1][1].title.set_text("Waterway")

    axes[1][2].imshow(input[7], cmap="gray")
    axes[1][2].title.set_text("Standing water")

    axes[1][3].imshow(input[8], cmap="gray")
    axes[1][3].title.set_text("Large vehicle")

    axes[1][4].imshow(input[9], cmap="gray")
    axes[1][4].title.set_text("Small vehicle")
    plt.savefig(name)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
  
config = configparser.ConfigParser()
config.read("hTorch/experiments/constants.cfg")

DATA_SIZE_TRAIN = config.getint("dataset", "data_size_train")
DATA_SIZE_VAL = config.getint("dataset", "data_size_val")
BATCH_SIZE = config.getint("dataset", "batch_size")
SHUFFLE = config.getboolean("dataset", "shuffle")

NUM_EPOCHS = config.getint("training", "num_epochs")
LEARNING_RATE = config.getfloat("training", "learning_rate")

ALPHA_AUX = config.getfloat("training", "num_epochs")

ALPHA = config.getfloat("loss", "alpha")
BETA = config.getfloat("loss", "beta")
GAMMA = config.getfloat("loss", "gamma")

IoU = IoU(num_classes=2)
trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]

bce =  torch.nn.BCEWithLogitsLoss()#alpha=ALPHA, beta=BETA, gamma=GAMMA)

# define file name with configs for saving the model
def get_short_name(name):
    split_name = str(name).split("_")
    short_name = ""
    if len(split_name) == 1:
        short_name = split_name[0][:2]
    else:
        short_name = "".join([name[0] for name in split_name])
    return short_name


def main():
    device = "cuda"
    if args.model == "psp":
        from models.qpsp import PSPNet
        model = PSPNet(quaternion=args.quaternion, loss=bce).to(device)
    if args.model == "swin":
        from models.qswin import SwinTransformer
        model = SwinTransformer(quaternion=args.quaternion).to(device)
    if args.model == "unet":
        from models.qunet import UNet
        model = UNet(quaternion=args.quaternion).to(device)

    resume = 0
    if args.checkpoint_weight_path:
        model.load_state_dict(torch.load(args.checkpoint_weight_path))
        resume = int(re.findall("(?<=e_)\d+", args.checkpoint_weight_path)[0])+1

    config_short_name = ""
    for section in config:
        section_dict = config[section]
        zipped = list(zip(section_dict.keys(), section_dict.values()))
        tmp = [name for tup in zipped for name in tup]
        for field in tmp:
            if field.replace(".", "").isdigit():
                field = str(field)
            else:
                field = get_short_name(field)

            config_short_name += field
    config_short_name += get_short_name(model.__class__.__name__)
    if args.quaternion:
        config_short_name += "_q"
    print(">" * 10, " parameters ", "<" * 10, "\n", config_short_name, sep="")
    print("saving to: ", args.save_dir)

    # class empirical sigmoid thresholds
    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    img, msk = stick_all_train()

    if not args.test:
        optimizer = MADGRAD(model.parameters(), lr=LEARNING_RATE)
        if args.checkpoint_optim_path:
            optimizer.load_state_dict(torch.load(args.checkpoint_optim_path))

        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)
        
        x_train, y_train = get_patches(img, msk, 3000)


        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, pin_memory=True,
                                         num_workers=0, drop_last=True)
       
        x_val, y_val = get_patches(img, msk, 1000)
        val = torch.utils.data.TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
        val_loader = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=SHUFFLE, pin_memory=True,
                                         num_workers=0, drop_last=True)

        dset_loaders = {"train": train_loader, "val": val_loader}
        dset_sizes = {"train": DATA_SIZE_TRAIN // BATCH_SIZE, "val": DATA_SIZE_VAL // BATCH_SIZE}
        for epoch in range(NUM_EPOCHS):

            print('-' * 40)
            print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode

                running_loss = 0.0
                running_metric_iou = 0.0
                total = 0
                # Iterate over data.
                for data in tqdm(dset_loaders[phase]):
                    # get the inputs
                    inputs, labels = data
                    
                    # inputs = 2*inputs -1
                    inputs, labels = inputs.to(device).float(), labels.to(device).float()
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    if phase == "train":
                        if args.model == "psp":
                            outputs, main_loss, aux_loss = model(inputs, labels)
                            loss = main_loss + ALPHA_AUX * aux_loss
                        else:
                            outputs = model(inputs)
                            loss = bce(outputs, labels)

                    else:
                        with torch.no_grad():
                            if args.model == "psp":
                                outputs = model(inputs, labels)
                                loss = bce(outputs, labels)
                            else:
                                outputs = model(inputs)
                                
                                loss = bce(outputs, labels)

                    total += labels.size(0)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    preds = torch.sigmoid(outputs).detach().cpu()
                    for i in range(10):
                        preds[:, i, ...] = (preds[:, i, ...] > trs[i])

                    iou = IoU(preds, labels.detach().cpu().long())
                    running_metric_iou += iou.detach().item()
                    
                    running_loss += loss.detach().item()


                epoch_loss = running_loss / total
                if phase == 'val':
                        lr_scheduler.step(epoch_loss)
                epoch_iou = running_metric_iou / total
                
                print('{} Loss: {:.4f} IoU: {:.4f}'.format(
                    phase, epoch_loss, epoch_iou))

                with open(os.path.join(args.save_dir, f"log_{phase[:2]}_loss_" + config_short_name + ".txt"), "a") as f:
                    f.write("%s\n" % epoch_loss)

                with open(os.path.join(args.save_dir, f"log_{phase[:2]}_iou_" + config_short_name + ".txt"), "a") as f:
                    f.write("%s\n" % epoch_iou)

            if args.save_last and epoch+resume != 0:
                os.remove(glob.glob(os.path.join(args.save_dir, f"weight_e_{epoch+resume-1}*"))[0])
                os.remove(glob.glob(os.path.join(args.save_dir, f"optim_e_{epoch+resume-1}*"))[0])

            torch.save(model.state_dict(), os.path.join(args.save_dir, f"weight_e_{epoch+resume}_" + config_short_name))
            torch.save(optimizer.state_dict(), os.path.join(args.save_dir, f"optim_e_{epoch+resume}_" + config_short_name))


        del x_train, y_train, train, train_loader
        x_train, y_train = get_patches(img, msk, 3000)
        train = torch.utils.data.TensorDataset(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=SHUFFLE, pin_memory=True,
                                        num_workers=0, drop_last=True)

        print()

    else:

        x_test = np.stack(np.split(np.stack(np.split(img, 25)), 25, axis = 2)).reshape(25*25, 167, 167, 8).astype(np.float16)
        del img
        y_test = np.squeeze(np.stack(np.split(np.stack(np.split(msk, 25)), 25, axis = 2)).reshape(25*25, 167, 167, 10)).astype(np.float16)
        del msk
        print(x_test.shape)
        test = torch.utils.data.TensorDataset(torch.from_numpy(np.transpose(x_test[:,:160, :160, :], (0,3,1,2))), torch.from_numpy(np.transpose(y_test[:,:160, :160, :], (0,3,1,2))))
        test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=SHUFFLE, pin_memory=True,
                                         num_workers=0, drop_last=True)        
        model.train(False)

        test_metric_iou = 0.0
        test_metric_f1 = 0.0
        test_metric_f1_crf = 0.0
        total = 0

        avg, trs = [], []
        for data in tqdm(test_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device).float(), labels.to(device).float()

            with torch.no_grad():
                if args.model == "psp":
                    outputs = model(inputs, labels)
                    loss = bce(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = bce(outputs, labels)

            preds = torch.sigmoid(outputs).detach().cpu()
            class_jk = []
            m, b_tr = 0, 0

            for i in range(10):
              for j in range(10):
                  tr = j / 10.0
                  pred_binary_mask = preds[:, i, ...] > tr

                  jk = jaccard_similarity_score(pred_binary_mask.detach().cpu().long().numpy().reshape(-1,10), labels[:, i, ...].detach().cpu().long().numpy().reshape(-1, 10))
                  if jk > m:
                      m = jk
                      b_tr = tr

              class_jk.append(m)

            avg.append(class_jk)
            trs.append(b_tr)

            total += labels.size(0)
        test_iou = np.mean(np.array(avg), axis = 0)
        test_iou_tot = np.mean(test_iou)

        print('Test IoU: {} Avg IoU: {}'.format(list(test_iou), test_iou_tot))

        with open(os.path.join(args.save_dir, "log_te_iou_" + config_short_name + ".txt"), "w+") as f:
            f.write("\n".join(["Per-Class IoU", *list(map(str, list(test_iou))), "Average IoU", str(test_iou_tot)]))


    pred, img, target = predict_id('6120_2_3', model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])

    plot_fig(pred.astype(np.float32), args.save_dir + "/pred_test")
    # plot_fig(target.astype(np.float32), args.save_dir + "/groundtruth_test")    
    plt.figure(figsize=[20,20])
    plt.imshow(to_rgb(np.transpose(img, (2,0,1)).astype(np.float32)))
    plt.savefig(args.save_dir + "/image")

    

if __name__ == '__main__':
    main()