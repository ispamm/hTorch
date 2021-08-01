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

sys.path.append('hTorch/')
sys.path.append('pytorch-image-models/')

from dstl_dataset import get_loader
from madgrad import MADGRAD

parser = argparse.ArgumentParser(description='htorch training and testing')
parser.add_argument('-m', '--model', help='model to train, choose from: {psp, swin, unet}', required=True)
parser.add_argument('-q', '--quaternion', help='wheter to use quaternions', action='store_true', default=False)
parser.add_argument('-s', '--save-dir', help='where to save checkpoint files', required=True)
parser.add_argument('-w', '--checkpoint-weight-path', help='saved checkpoint weight file to resume trainng from')
parser.add_argument('-o', '--checkpoint-optim-path', help='saved checkpoint optimizer to resume trainng from')
parser.add_argument('-t', '--test', help='test mode', action='store_true', default=False)
parser.add_argument('-l', '--save-last', help='save only last epoch', action='store_true', default=False)

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

config = configparser.ConfigParser()
config.read("hTorch/experiments/constants.cfg")

DATA_SIZE_TRAIN = config.getint("dataset", "data_size_train")
DATA_SIZE_VAL = config.getint("dataset", "data_size_val")
BATCH_SIZE = config.getint("dataset", "batch_size")
NUM_EPOCHS = config.getint("training", "num_epochs")
LEARNING_RATE = config.getfloat("training", "learning_rate")

ALPHA_AUX = config.getfloat("training", "num_epochs")

ALPHA = config.getfloat("loss", "alpha")
BETA = config.getfloat("loss", "beta")
GAMMA = config.getfloat("loss", "gamma")

IoU = IoU(num_classes=2)
tversky =  TverskyLoss(alpha=ALPHA, beta=BETA)

def tversky_focal(*args):
    return tversky(*args)**GAMMA

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
        model = PSPNet(quaternion=args.quaternion, loss=tversky_focal).to(device)
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

    # class empirical sigmoid thresholds
    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]

    if not args.test:
        train_loader, val_loader = get_loader("train", BATCH_SIZE), get_loader("val", 2)
        optimizer = MADGRAD(model.parameters(), lr=LEARNING_RATE)
        if args.checkpoint_optim_path:
            optimizer.load_state_dict(torch.load(args.checkpoint_optim_path))

        lr_scheduler = ReduceLROnPlateau(optimizer, 'min')

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
                    inputs, labels = inputs.to(device), labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    if phase == "train":
                        if args.model == "psp":
                            outputs, main_loss, aux_loss = model(inputs, labels)
                            loss = main_loss + ALPHA_AUX * aux_loss
                        else:
                            outputs = model(inputs)
                            loss = tversky_focal(outputs, labels.argmax(1))

                    else:
                        with torch.no_grad():
                            if args.model == "psp":
                                outputs = model(inputs, labels)
                                loss = tversky_focal(outputs, labels.argmax(1))
                            else:
                                outputs = model(inputs)
                                loss = tversky_focal(outputs, labels.argmax(1))

                    # statistics
                    preds = torch.sigmoid(outputs).detach().cpu()
                    for i in range(10):
                        preds[:, i, ...] = (preds[:, i, ...] > trs[i])

                    iou = IoU(preds, labels.detach().cpu())
                    running_metric_iou += iou.detach().item()

                    total += labels.size(0)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    if phase == 'val':
                        lr_scheduler.step(loss)
                    
                    running_loss += loss.detach().item()


                epoch_loss = running_loss / total
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

            print()

    else:
        test_loader = get_loader("test", 2)
        model.train(False)

        test_metric_iou = 0.0
        test_metric_f1 = 0.0
        test_metric_f1_crf = 0.0
        total = 0
        for data in tqdm(test_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                if args.model == "psp":
                    outputs = model(inputs, labels)
                    loss = tversky_focal(outputs, labels.argmax(1))
                else:
                    outputs = model(inputs)
                    loss = tversky_focal(outputs, labels.argmax(1))

            preds = torch.sigmoid(outputs).detach().cpu()
            # for i in range(10):
            #     preds[:, i, ...] = (preds[:, i, ...] > trs[i])

            iou = IoU(preds, labels.detach().cpu())
            test_metric_iou += iou.detach().item()

            total += labels.size(0)

        test_iou = test_metric_iou / total

        print('Test IoU: {:.4f}'.format(test_iou))

        with open(os.path.join(args.save_dir, "log_te_iou_" + config_short_name + ".txt"), "a") as f:
            f.write("%s\n" % test_iou)


    print()

    plt.figure(figsize=[10, 10])
    plt.imshow(preds[0][3].detach().cpu().numpy())
    plt.savefig("test.jpg")

    plt.figure(figsize=[10, 10])
    plt.imshow(labels[0][3].detach().cpu().numpy())
    plt.savefig("ground_truth.jpg")


if __name__ == '__main__':
    main()
