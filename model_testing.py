from torchsummary import summary
from models.ModifiedAttentionTransformer import ModifiedAttentionTransformer
import torch
from accelerate import Accelerator
from data_loader import data_loader
from models.modules.weight_initialization import custom_weight_initialization
from utils.logger import Logger
from models.modules.loss_functions import loss_function, mae_loss, mean_error, mean_absolute_error
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import yaml


################# Model Name ####################
model_name = "ModifiedAttentionTransformer"
model_number = 13
saved_model_name = f"{model_name}_{model_number}_epoch-None.pt"
path = f"models/{model_name}/saved_models/{model_name}_{model_number}/{saved_model_name}"
#################################################

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
gen = torch.manual_seed(1000)
torch.manual_seed(1000)

with open(f"models/{model_name}/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Import Model
if model_name == "ModifiedAttentionTransformer":
    from models.ModifiedAttentionTransformer import ModifiedAttentionTransformer
    model = ModifiedAttentionTransformer.TransformerForSignalEstimation(config).float().to(device)
elif model_name == "ResNet":
    from models.ResNet import ResNet
    model = ResNet.ResNet(config).float().to(device)
elif model_name == "DenseNet":
    from models.DenseNet import DenseNet
    model = DenseNet.DenseNet(config).float().to(device)
else:
    raise ValueError(f"Model {model_name} is not implemented.")

model.load_state_dict(torch.load(path, map_location=device, weights_only=True))


train_dataset, _, train_loader, _ = data_loader.data_loader(
    batch_size=128,
    shuffle=True,
    val_split=1.0,
    generator=gen,
    mode="test",
    # model_dir=f"models/{model_name}/saved_models/{model_name}_{model_number}",
)

model.eval()

val_loss = 0.0
sbp_errors = []
dbp_errors = []
waveform_mae_errors = []
sbp_mean_errors = []
dbp_mean_errors = []
waveform_me_errors = []
true_abp_values = []
estimated_abp_values = []

with torch.no_grad():
    index = 0
    for abp, ppg, features in train_loader:
        index += 1
        print(f"{index} / {len(train_loader)}", end="\r")
        abp, ppg, features = abp.to(torch.float32), ppg.to(torch.float32), features.to(torch.float32)
        abp, ppg, features = abp.to(device), ppg.to(device), features.to(device)
        estimated_abp = model(ppg, features)
        estimated_abp = estimated_abp.reshape(estimated_abp.shape[0], estimated_abp.shape[1])
        loss = mae_loss(estimated_abp, abp)
        true_abp_values.append(abp)
        estimated_abp_values.append(estimated_abp)
        me = mean_error(estimated_abp, abp)
        mae = mean_absolute_error(estimated_abp, abp)

        sbp = torch.max(abp, 1).values.tolist()
        dbp = torch.min(abp, 1).values.tolist()
        _map = (torch.sum(abp, 1) / len(abp)).tolist()
        sbp_estimated = torch.max(estimated_abp, 1).values.tolist()
        dbp_estimated = torch.min(estimated_abp, 1).values.tolist()
        _map_estimated = (torch.sum(estimated_abp, 1) / len(estimated_abp)).tolist()

        mae_sbp = np.abs(np.subtract(sbp, sbp_estimated))
        mae_dbp = np.abs(np.subtract(dbp, dbp_estimated))
        mae_map = np.abs(np.subtract(_map, _map_estimated))

        me_sbp = np.subtract(sbp, sbp_estimated)
        me_dbp = np.subtract(dbp, dbp_estimated)
        me_map = np.subtract(_map, _map_estimated)

        for i in range(len(mae_sbp)):
            sbp_errors.append(mae_sbp[i])
            dbp_errors.append(mae_dbp[i])
            waveform_mae_errors.append(mae[i].item())
            sbp_mean_errors.append(me_sbp[i])
            dbp_mean_errors.append(me_dbp[i])
            waveform_me_errors.append(me[i].item())
        
        val_loss += loss.item()

print(f"sbp mae errors mean: {np.mean(sbp_errors)}")
print(f"dbp mae errors mean: {np.mean(dbp_errors)}")
print(f"waveform mae errors mean: {np.mean(waveform_mae_errors)}")

print(f"sbp me errors mean: {np.mean(sbp_mean_errors)}")
print(f"dbp me errors mean: {np.mean(dbp_mean_errors)}")
print(f"waveform me errors mean: {np.mean(waveform_me_errors)}")

print(f"sbp mae errors std: {np.std(sbp_errors)}")
print(f"dbp mae errors std: {np.std(dbp_errors)}")
print(f"waveform mae errors std: {np.std(waveform_mae_errors)}")

print(f"sbp me errors std: {np.std(sbp_mean_errors)}")
print(f"dbp me errors std: {np.std(dbp_mean_errors)}")
print(f"waveform me errors std: {np.std(waveform_me_errors)}")
