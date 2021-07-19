import torch
import argparse
import configparser
from .dstl_dataset import get_loader

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m', '--model', help='model to train, choose from: {psp, swin, unet}')
parser.add_argument('-q', '--quaternion', help='wheter to use quaternions', action='store_true')
args = parser.parse_args()

if args.model == "psp":
    from .models.qpsp import PSPNet
    model = PSPNet(quaternion = args.quaternion)
if args.model == "swin":
    from .models.qswin import SwinTransformer
    model = SwinTransformer(quaternion = args.quaternion)
if args.model == "unet":
    from .models.unet import UNet
    model = UNet(quaternion = args.quaternion)

config = configparser.SafeConfigParser()
config.read("hTorch/experiments/constants.cfg")

DATA_SIZE_TRAIN = config.getint("dataset", "data_size_train")
DATA_SIZE_VAL = config.getint("dataset", "data_size_val")
BATCH_SIZE = config.getint("dataset", "batch_size")
NUM_EPOCHS = config.getfloat("training", "num_epochs")

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

config_short_name = ""
for section in config:
    section_dict = config[section]
    zipped = list(zip(section_dict.keys(), section_dict.values()))
    tmp = [name for tup in zipped for name in tup]
    for field in tmp:
        if field.replace(".","").isdigit():
            field = str(field)
        else:
            field = get_short_name(field)
            
        config_short_name += field 

config_short_name += get_short_name(model.__class__.__name__)
print(">"*10, "parameters", "<"*10, "\n", config_short_name, sep=" ")

train_loader, val_loader = get_loader("train"), get_loader("val")

trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
dset_loaders = {"train":train_loader, "val":val_loader}
dset_sizes = {"train":DATA_SIZE_TRAIN//BATCH_SIZE, "val":DATA_SIZE_VAL//BATCH_SIZE}
for epoch in range(NUM_EPOCHS):

    print('Epoch {}/{}'.format(epoch, NUM_EPOCHS))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_metric = 0.0
        running_metric_crf = 0.0
        running_metric_f1 = 0.0
        total = 0
        # Iterate over data.
        for step in tqdm(range(dset_sizes[phase])):
          for data in dset_loaders[phase]:
              # get the inputs
              inputs, labels = data
              # zero the parameter gradients
              optimizer.zero_grad()

              # forward
              if phase == "train":
                  outputs, main_loss, aux_loss, f1 = model(inputs.to(device), labels.to(device))
                  loss = main_loss + ALPHA_AUX * aux_loss
              else:
                  with torch.no_grad():
                    outputs = model(inputs.to(device), labels.to(device))
                    loss, f1 = criterion(outputs.float(), labels.to(device).float())

              if phase == 'train':
                  loss.backward()                
                  optimizer.step()
                  # lr_scheduler.step()

              # statistics
              running_loss += float(loss)
              preds = torch.sigmoid(outputs).detach().cpu()
              for i in range(10):
                  preds[:, i, ...] = (preds[:, i, ...] > trs[i])
                          
              probs = torch.sigmoid(preds).data.cpu().numpy()
              crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
              
              for i in range(10):
                  crf[:, i, ...] = (crf[:, i, ...] > trs[i])

              score = jaccard_coef(labels.detach().cpu(), preds)
              score_crf = jaccard_coef(labels.detach().cpu(), torch.from_numpy(crf))

              running_metric_f1 += f1
              running_metric_crf += score_crf
              running_metric += score
              total += labels.size(0)

  
        epoch_loss = running_loss / total
        epoch_acc = running_metric / total
        epoch_acc_crf = running_metric_crf / total
        epoch_acc_f1 = running_metric_f1 / total
        print('{} Loss: {:.4f} Acc: {:.4f} Acc crf: {:.4f} F1: {:.4f}'.format(
            phase, epoch_loss, epoch_acc.numpy().round(2), 
            epoch_acc_crf.numpy().round(2),epoch_acc_f1.cpu().detach().numpy().round(2)))

        # deep copy the model
        with open('{}_losses_75.txt'.format(phase), "a") as f:
            f.write("\n%s" % epoch_loss)
            f.close()

        with open('{}_metrics_75.txt'.format(phase), "a") as f:
            f.write("\n%s" % epoch_acc)
            f.close()

        torch.save(model.state_dict(),  "/content/drive/MyDrive/QPSP/Quaternion PSPNet DSTL_Mod_75_"+ str(epoch+1))
        torch.save(optimizer.state_dict(),  "/content/drive/MyDrive/QPSP/Quaternion PSPNet DSTL_Mod_optim_75_"+ str(epoch+1))
        print()
   
   
   
   
    # def configure_optimizers(self):
    #     optimizer = MADGRAD(self.parameters(), lr=LEARNING_RATE)
    #     return optimizer

    # def focal_tversky_loss(self, x, y):
    #     loss = FocalTverskyLoss()(x, y)
    #     return loss

    # def training_step(self, train_batch, batch_idx):
    #     inputs, labels = train_batch
    #     outputs, main_loss, aux_loss = self.forward(inputs, labels)

    #     probs = torch.sigmoid(outputs).data.cpu().numpy()
    #     crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
    #     crf = np.ascontiguousarray(crf)
    #     f1_crf = f1_score(torch.from_numpy(crf).to(self.device), labels)

    #     loss = main_loss + ALPHA_AUX * aux_loss
    #     f1 = f1_score(outputs, labels)

    #     self.log('train_loss', loss)
    #     self.log('train_f1', f1)
    #     self.log('train_f1_crf', f1_crf)
    #     return loss

    # def validation_step(self, val_batch, batch_idx):
    #     inputs, labels = val_batch
    #     outputs = self.forward(inputs, labels)

    #     probs = torch.sigmoid(outputs).data.cpu().numpy()
    #     crf = np.stack(list(map(dense_crf_wrapper, zip(inputs.cpu().numpy(), probs))))
    #     crf = np.ascontiguousarray(crf)
    #     f1_crf = f1_score(torch.from_numpy(crf).to(self.device), labels)

    #     loss = self.focal_tversky_loss(outputs.float(), labels.float())
    #     f1 = f1_score(outputs, labels)

    #     self.log('val_loss', loss)
    #     self.log('val_f1_crf', f1_crf)
    #     self.log('val_f1', f1)

