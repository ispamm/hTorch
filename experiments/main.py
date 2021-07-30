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

sys.path.append('hTorch/')
sys.path.append('pytorch-image-models/')

from dstl_dataset import get_loader
from madgrad import MADGRAD
from loss import FocalTverskyLoss
from utils import f1_score, jaccard_score, to_rgb
from crf import dense_crf_wrapper

parser = argparse.ArgumentParser(description='htorch training and testing')
parser.add_argument('-m', '--model', help='model to train, choose from: {psp, swin, unet}', required=True)
parser.add_argument('-q', '--quaternion', help='wheter to use quaternions', action='store_true', default=False)
parser.add_argument('-s', '--save-dir', help='where to save checkpoint files', required=True)
parser.add_argument('-w', '--checkpoint-weight-path', help='saved checkpoint weight file to resume trainng from')
parser.add_argument('-o', '--checkpoint-optim-path', help='saved checkpoint optimizer to resume trainng from')
parser.add_argument('-t', '--test', help='test mode', action='store_true', default=False)
parser.add_argument('-l', '--save-last', help='save only last epoch', action='store_true', default=False)

args = parser.parse_args()

config = configparser.ConfigParser()
config.read("hTorch/experiments/constants.cfg")

DATA_SIZE_TRAIN = config.getint("dataset", "data_size_train")
DATA_SIZE_VAL = config.getint("dataset", "data_size_val")
BATCH_SIZE = config.getint("dataset", "batch_size")
NUM_EPOCHS = config.getint("training", "num_epochs")
LEARNING_RATE = config.getfloat("training", "learning_rate")

ALPHA_AUX = config.getfloat("training", "num_epochs")


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
        model = PSPNet(quaternion=args.quaternion).to(device)
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
        criterion = FocalTverskyLoss()

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
                running_metric_f1 = 0.0
                running_metric_f1_crf = 0.0
                total = 0
                # Iterate over data.
                for data in tqdm(dset_loaders[phase]):
                    # get the inputs
                    inputs, labels = data
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
                            loss = criterion(outputs, labels)

                    else:
                        with torch.no_grad():
                            if args.model == "psp":
                                outputs = model(inputs, labels)
                            else:
                                outputs = model(inputs)

                    # statistics
                    running_loss += loss.detach().item()
                    preds = torch.sigmoid(outputs).detach().cpu()
                    for i in range(10):
                        preds[:, i, ...] = (preds[:, i, ...] > trs[i])

                    probs = torch.sigmoid(preds).data.cpu().numpy()
                    crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))

                    for i in range(10):
                        crf[:, i, ...] = (crf[:, i, ...] > trs[i])

                    iou = jaccard_score(labels.detach().cpu(), preds)
                    f1 = f1_score(outputs, labels)
                    f1_crf = f1_score(labels.detach().cpu(), torch.from_numpy(crf).contiguous())

                    running_metric_iou += iou.detach().item()
                    running_metric_f1 += f1.detach().item()
                    running_metric_f1_crf += f1_crf.detach().item()

                    total += labels.size(0)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step(f1)

                epoch_loss = running_loss / total
                epoch_iou = running_metric_iou / total
                epoch_f1 = running_metric_f1 / total
                epoch_f1_crf = running_metric_f1_crf / total

                print('{} Loss: {:.4f} IoU: {:.4f} F1: {:.4f} F1 crf: {:.4f}'.format(
                    phase, epoch_loss, epoch_iou,
                    epoch_f1, epoch_f1_crf))

                with open(os.path.join(args.save_dir, f"log_{phase[:2]}_loss_" + config_short_name + ".txt"), "a") as f:
                    f.write("%s\n" % epoch_loss)

                with open(os.path.join(args.save_dir, f"log_{phase[:2]}_iou_" + config_short_name + ".txt"), "a") as f:
                    f.write("%s\n" % epoch_iou)

                with open(os.path.join(args.save_dir, f"log_{phase[:2]}_f1_" + config_short_name + ".txt"), "a") as f:
                    f.write("%s\n" % epoch_f1)

                with open(os.path.join(args.save_dir, f"log_{phase[:2]}_f1crf_" + config_short_name + ".txt"),
                          "a") as f:
                    f.write("%s\n" % epoch_f1_crf)

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
                else:
                    outputs = model(inputs)

            preds = torch.sigmoid(outputs).detach().cpu()
            for i in range(10):
                preds[:, i, ...] = (preds[:, i, ...] > trs[i])

            probs = torch.sigmoid(preds).data.cpu().numpy()
            crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))

            for i in range(10):
                crf[:, i, ...] = (crf[:, i, ...] > trs[i])

            iou = jaccard_score(labels.detach().cpu(), preds)
            f1 = f1_score(outputs, labels)
            f1_crf = f1_score(labels.detach().cpu(), torch.from_numpy(crf).contiguous())

            test_metric_iou += iou.detach().item()
            test_metric_f1 += f1.detach().item()
            test_metric_f1_crf += f1_crf.detach().item()

            total += labels.size(0)

        test_iou = test_metric_iou / total
        test_f1 = test_metric_f1 / total
        test_f1_crf = test_metric_f1_crf / total

        print('Test IoU: {:.4f} F1: {:.4f} F1 crf: {:.4f}'.format(
            test_iou,
            test_f1, test_f1_crf))

        with open(os.path.join(args.save_dir, "log_te_iou_" + config_short_name + ".txt"), "a") as f:
            f.write("%s\n" % test_iou)

        with open(os.path.join(args.save_dir, "log_te_f1_" + config_short_name + ".txt"), "a") as f:
            f.write("%s\n" % test_f1)

        with open(os.path.join(args.save_dir, "log_te_f1crf_" + config_short_name + ".txt"), "a") as f:
            f.write("%s\n" % test_f1_crf)

    print()

    plt.figure(figsize=[10, 10])
    plt.imshow(to_rgb(outputs[0].detach().cpu().numpy()))
    plt.savefig("test.jpg")


if __name__ == '__main__':
    main()
