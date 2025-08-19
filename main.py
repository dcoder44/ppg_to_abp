from torchsummary import summary
import torch
from accelerate import Accelerator
from data_loader import data_loader
from utils.logger import Logger
from models.modules.loss_functions import mae_loss
import yaml

################# Model Name ####################
model_name = "ModifiedAttentionTransformer"
model_number = 13
description = "MixFFN + PE."
#################################################

accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device
gen = torch.manual_seed(1000)
torch.manual_seed(1000)

# Import Model
if model_name == "ModifiedAttentionTransformer":
    from models.ModifiedAttentionTransformer import ModifiedAttentionTransformer
    with open(f"models/{model_name}/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    model = ModifiedAttentionTransformer.TransformerForSignalEstimation(config).float().to(device)
elif model_name == "ResNet":
    from models.ResNet import ResNet
    with open(f"models/{model_name}/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    model = ResNet.ResNet(config).float().to(device)
elif model_name == "DenseNet":
    from models.DenseNet import DenseNet
    with open(f"models/{model_name}/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    model = DenseNet.DenseNet(config).float().to(device)
else:
    raise ValueError(f"Model {model_name} is not implemented.")

# Data Loader
train_dataset, val_dataset, train_loader, val_loader = data_loader.data_loader(
    batch_size=config.get("batch_size", 64),
    shuffle=True,
    val_split=config.get("val_split", 0.9),
    generator=gen,
    mode=config.get("mode", "train"),
    # model_dir=f"models/{model_name}/saved_models/{model_name}_{model_number}"  # Uncomment if use_features is True
    )

train_losses = []
val_losses = []

optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 0.0001))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, threshold=0.001)

train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, val_loader, model, optimizer, scheduler)


# Save model details
print("Saving model details...")
logger = Logger(model_name, model_number)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.save_model_details(
    f"{description}" \
    f"\nTotal parameters: {total_params}, Trainable parameters: {trainable_params}",
    model_parameters=config
)

print("Model details saved. Starting training...") 
min_train_loss = float('inf')
min_val_loss = float('inf')   
best_epoch = 0
num_epochs = config.get("num_epochs", 100)
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0

    for i, (abp, ppg, features) in enumerate(train_loader):
        with accelerator.autocast():
            abp, ppg, features = abp.to(torch.float32), ppg.to(torch.float32), features.to(torch.float32)
            abp, ppg, features = abp.to(device), ppg.to(device), features.to(device)
            optimizer.zero_grad()
            abp_predicted = model(ppg, features)
            abp_predicted = abp_predicted.reshape(abp_predicted.shape[0], abp_predicted.shape[1])
            loss = mae_loss(abp_predicted, abp)
            max_norm = 2.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            train_loss += loss.item()
            accelerator.backward(loss)
            optimizer.step()

            print(f"Step: {i+1} / {len(train_loader)}, loss: {train_loss / (i+1)}", end='\r')

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Evaluation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for abp, ppg, features in val_loader:
            with accelerator.autocast():
                abp, ppg, features = abp.to(torch.float32), ppg.to(torch.float32), features.to(torch.float32)
                abp, ppg, features = abp.to(device), ppg.to(device), features.to(device)
                abp_predicted = model(ppg, features)
                abp_predicted = abp_predicted.reshape(abp_predicted.shape[0], abp_predicted.shape[1])
                loss = mae_loss(abp_predicted, abp)
                val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(train_loss)

    print(f"epoch {epoch + 1}/{num_epochs}, loss: {train_loss:.4f}, val_loss: {val_loss:.4f} current_lr: {scheduler.get_last_lr()[-1]: 0.2e}")
    
    if train_loss < min_train_loss and val_loss < min_val_loss:
        min_train_loss = train_loss
        min_val_loss = val_loss
        best_epoch = epoch + 1
        logger.save_model_and_loss(model, None, train_losses, val_losses)
        
logger.save_model_and_loss(model, num_epochs, train_losses, val_losses)
logger.save_test_results(results=f"\n\nBest model found at epoch {best_epoch} with train loss {min_train_loss:.4f} and val loss {min_val_loss:.4f}")
print(f"Best model found at epoch {best_epoch} with train loss {min_train_loss:.4f} and val loss {min_val_loss:.4f}")

