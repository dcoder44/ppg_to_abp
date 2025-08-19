from torchsummary import summary
import torch
from accelerate import Accelerator
from data_loader import data_loader
from utils.logger import Logger
from models.modules.loss_functions import mae_loss, mean_error, mean_absolute_error
import yaml
import numpy as np

################## Model Name ####################
model_name = "ResLSTM"
model_number = 1
description = "ResLSTM. 5-fold cross validation."
n_splits = 5
##################################################


accelerator = Accelerator(mixed_precision="fp16")
device = accelerator.device

with open(f"models/{model_name}/config.yaml", "r") as file:
    config = yaml.safe_load(file)

seed = config.get("seed", 1000)
gen = torch.manual_seed(seed)
torch.manual_seed(seed)

fold_loaders = data_loader.get_nfold_data_loaders(
    n_splits=config.get("n_splits", 5),
    batch_size=config.get("batch_size", 64),
    shuffle=True,
    mode="n-fold",
    max_samples=config.get("max_samples", None))


min_train_loss = float('inf')
min_val_loss = float('inf')
best_epoch, best_fold = 0, 0
for fold, (train_loader, val_loader, test_loader) in enumerate(fold_loaders):
    print(f"Training on fold {fold+1}/{n_splits}...")
    if model_name == "ModifiedAttentionTransformer":
        from models.ModifiedAttentionTransformer import ModifiedAttentionTransformer
        model = ModifiedAttentionTransformer.TransformerForSignalEstimation(config).float().to(device)
    elif model_name == "DenseNet":
        from models.DenseNet import DenseNet
        model = DenseNet.DenseNet(config).float().to(device)
    elif model_name == "ResNet":
        from models.ResNet import ResNet
        model = ResNet.ResNet(config).float().to(device)
    elif model_name == "CNNLSTM":
        from models.CNNLSTM import CNNLSTM
        model = CNNLSTM.CNNLSTM(config).float().to(device)
    elif model_name == "ResLSTM":
        from models.ResLSTM import ResLSTM
        model = ResLSTM.ResLSTM(config).float().to(device)
    else:
        raise ValueError(f"Model {model_name} is not implemented.")
    
    if fold == 0:
        print("Saving model details...")
        logger = Logger(model_name, model_number)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.save_model_details(
            f"{description} \n" \
            f"Total parameters: {total_params}, Trainable parameters: {trainable_params}",
            model_parameters=config
        )
    
    train_losses = []
    val_losses = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 0.0001))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, threshold=0.001)
    train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(train_loader, val_loader, model, optimizer, scheduler)
    
    num_epochs = config.get("num_epochs", 50)
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

        print(f"fold {fold+1}/{len(fold_loaders)}, epoch {epoch + 1}/{num_epochs}, loss: {train_loss:.4f}, val_loss: {val_loss:.4f} current_lr: {scheduler.get_last_lr()[-1]: 0.2e}")

        if train_loss < min_train_loss and val_loss < min_val_loss:
            min_train_loss = train_loss
            min_val_loss = val_loss
            best_epoch = epoch + 1
            best_fold = fold + 1
            logger.save_cv_model_and_loss(model, fold+1, train_losses, val_losses)
        else:
            logger.save_cv_model_and_loss(model, fold+1, train_losses, val_losses, save_model=False)
            
    logger.save_cv_model(model, fold+1)
    
    # Evaluation on the test set
    print(f"Evaluating on the test set for fold {fold+1}/{n_splits}...")
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
        for abp, ppg, features in test_loader:
            index += 1
            print(f"{index} / {len(test_loader)}", end="\r")
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

    logger.save_test_results(
        results=f"""\n\nFold {fold + 1} test results:
        sbp mae errors mean: {np.mean(sbp_errors)}
        dbp mae errors mean: {np.mean(dbp_errors)}
        waveform mae errors mean: {np.mean(waveform_mae_errors)}
        sbp me errors mean: {np.mean(sbp_mean_errors)}
        dbp me errors mean: {np.mean(dbp_mean_errors)}
        waveform me errors mean: {np.mean(waveform_me_errors)}
        sbp mae errors std: {np.std(sbp_errors)}
        dbp mae errors std: {np.std(dbp_errors)}
        waveform mae errors std: {np.std(waveform_mae_errors)}
        sbp me errors std: {np.std(sbp_mean_errors)}
        dbp me errors std: {np.std(dbp_mean_errors)}
        waveform me errors std: {np.std(waveform_me_errors)}
    """)

logger.save_test_results(results=f"\n\nBest model found at fold {best_fold}, epoch {best_epoch} with train loss {min_train_loss:.4f} and val loss {min_val_loss:.4f}")
print(f"Best model found at fold {best_fold}, epoch {best_epoch} with train loss {min_train_loss:.4f} and val loss {min_val_loss:.4f}")

