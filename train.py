import argparse
import torch
from torch.utils.data import DataLoader
from dataset import CREATISDataset, DIRLABDataset
from utils.yaml_reader import load_config
from models.model import FFCResNetGenerator
from models.SpatialTransformerNetwork import SpatialTransformation
from losses import compute_loss
# from inference import test_per_case
from tqdm import tqdm
from utils.view3d_image import *

parser = argparse.ArgumentParser()
parser.add_argument('--training_config_path', type=str, default='./configs/training_settings.yaml', help='path to the training config')
parser.add_argument('--model_config_path', type=str, default='./configs/FFCResnetGenerator_settings.yaml', help='path to the model config')
parser.add_argument('--exp', type=int, required=True, help='experiment number')
parser.add_argument('--seed', type=int, default=23)
parser.add_argument('--load_weights', type=int, default=0)

args = parser.parse_args()

torch.manual_seed(args.seed)
EXPERIMENT = args.exp

# Loading configs
train_configs = load_config(args.training_config_path)
model_configs = load_config(args.model_config_path)

# Creating datasets
train_dataset = CREATISDataset(root=train_configs['train_data_path'],
                            case_list=train_configs['train_cases'])


print('TRAIN DATA: {} pairs of fixed/moving images.'.format(len(train_dataset)))
train_loader = DataLoader(train_dataset, **train_configs['data_loader'])


# test_dataset = DIRLABDataset(root=train_configs['test_data_path'],
#                           case_list=train_configs['test_cases'],
#                           phases=[0, 5]) # maximum inhale and exhale
# print('TEST DATA: {} pairs of fixed/moving images.'.format(len(test_dataset)))
# test_loader = DataLoader(test_dataset, **train_configs['data_loader'])


# Model defining
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
FFCGenerator = FFCResNetGenerator(input_nc=model_configs['input_nc'],
                                  output_nc=model_configs['output_nc'],
                                  n_downsampling=model_configs['n_downsampling'],
                                  n_blocks=model_configs['n_blocks'],
                                  init_conv_kwargs=model_configs['init_conv_kwargs'],
                                  downsample_conv_kwargs=model_configs['downsample_conv_kwargs'],
                                  resnet_conv_kwargs=model_configs['resnet_conv_kwargs']).to(device)
stn = SpatialTransformation(use_gpu=False)


# optimization settings
optimizer = torch.optim.Adam(FFCGenerator.parameters(),
                            lr=train_configs['lr'])
                            

# Load the weights
if args.load_weights:
    FFCGenerator.load_state_dict(torch.load(train_configs['save_dir'] + "exp" + str(EXPERIMENT) + "_patch64_generator.pth"))
    stn.load_state_dict(torch.load(train_configs['save_dir'] + "exp" + str(EXPERIMENT) + "_patch64_stn.pth"))

# Loop over epochs for training
for epoch in range(train_configs['n_epochs']+1):

    batch_counter = 0
    epoch_train_loss = 0.0
    running_train_loss = 0.0
    for paired_patches in train_loader:
        batch_counter += 1

        
        # paired_patches shape: [batch, 2, d, h, w]   
        pair = paired_patches.to(device)
        DVF = FFCGenerator(pair).cpu()

        fi = torch.unsqueeze(pair[:, 0, :], 1).cpu() # fixed patch 
        mi = torch.unsqueeze(pair[:, 1, :], 1).cpu() # moving patch
        registered_images = stn(mi.permute(0, 1, 4, 3, 2), # (batch, c, w, h, d)
                                DVF.permute(0, 1, 4, 3, 2))

        registered_images = registered_images.permute(0, 1, 4, 3, 2)


        # compute loss
        loss = compute_loss(y_pred=registered_images.to(device),
                                y_true=fi.to(device),
                                dvf=DVF.to(device),
                                window_size=train_configs['window_size'],
                                lamda=train_configs['lamda'],
                                mu=train_configs['mu'])

        epoch_train_loss += loss.item()

        # import ipdb; ipdb.set_trace()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        del loss, pair, DVF, fi, mi, registered_images  # to reduce memory usage

    if epoch % train_configs['print_every'] == 0:
        print("Epoch: {}, train loss={}".format(epoch, epoch_train_loss / len(train_loader)))

    # if batch_counter % train_configs['save_every'] == 0:
    #     torch.save(FFCGenerator.state_dict(), train_configs['save_dir'] + "exp" + str(EXPERIMENT) + "_patch64_generator.pth")
    #     torch.save(stn.state_dict(), train_configs['save_dir'] + "exp" + str(EXPERIMENT) + "_patch64_stn.pth")

    #     for c in range(10):
    #         print("--------------")
    #         test_per_case(experiment=EXPERIMENT, case_num=c+1, phases_list=[0, 5], epoch_idx=epoch)

    #     print("RUNNING TRAIN LOSS={}".format(running_train_loss / train_configs['save_every']))
    #     running_train_loss = 0.0

    # validation
    # test_loss = 0.0
    # with torch.no_grad():
    #     for paired_patches, _ in tqdm(test_loader):

    #         # paired_patches shape: [batch, 2, d, h, w]   
    #         pair = paired_patches.to(device)
    #         DVF = FFCGenerator(pair).cpu()

    #         fi = torch.unsqueeze(pair[:, 0, :], 1).cpu() # fixed patch 
    #         mi = torch.unsqueeze(pair[:, 1, :], 1).cpu() # moving patch
    #         registered_images = stn(mi.permute(0, 1, 4, 3, 2), # (batch, c, w, h, d)
    #                                 DVF.permute(0, 1, 4, 3, 2))

    #         registered_images = registered_images.permute(0, 1, 4, 3, 2)

    #         # compute loss
    #         loss = compute_loss(y_pred=registered_images.to(device),
    #                                 y_true=fi.to(device),
    #                                 dvf=DVF.to(device),
    #                                 window_size=train_configs['window_size'],
    #                                 lamda=train_configs['lamda'],
    #                                 mu=train_configs['mu'])

    #         test_loss += registered_images.shape[0] * loss.item()
    #         del loss, pair, DVF, fi, mi, registered_images  # to reduce memory usage

    # torch.save(FFCGenerator.state_dict(), train_configs['save_dir'] + "exp" + str(EXPERIMENT) + "_patch64_generator.pth")
    # torch.save(stn.state_dict(), train_configs['save_dir'] + "exp" + str(EXPERIMENT) + "_patch64_stn.pth")

    # # test on test set
    # for c in range(5):
    #     print("--------------")
    #     test_per_case(experiment=EXPERIMENT, case_num=c+1, phases_list=[0, 5], epoch_idx=epoch, save_figs=True)

    # print("EPOCH {}/{}: TRAIN LOSS={}".format(epoch, train_configs['n_epochs'], epoch_train_loss / len(train_dataset)))
    # print("-----------  TEST LOSS={}".format(test_loss / len(test_dataset)))
    # print("-------------------------------------------------")
    

    




        





