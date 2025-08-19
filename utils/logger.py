import os
import numpy as np
import torch

class Logger:
    def __init__(self, model_name, model_number=1):
        self.models_folder_path = "models"
        self.folder_path = ""
        self.model_name = model_name
        self.model_number = model_number

    def save_model_details(self, model_summary, model_parameters):
        self.folder_path = f"{self.models_folder_path}/{self.model_name}/saved_models/{self.model_name}_{self.model_number}"
        os.makedirs(self.folder_path, exist_ok=True)
        model_details = f"{str(model_summary)} \n\n {str(model_parameters)}"
        with open(f"{self.folder_path}/model_summary.txt", "w", encoding="utf-8") as f:
            f.write(model_details)

    def save_model_and_loss(self, model, epoch, train_losses, val_losses):
        model_path = f"{self.folder_path}/{self.model_name}_{self.model_number}_epoch-{str(epoch)}.pt"
        train_loss_path = f"{self.folder_path}/{self.model_name}_{self.model_number}_train_loss.npy"
        val_loss_path = f"{self.folder_path}/{self.model_name}_{self.model_number}_val_loss.npy"
        torch.save(model.state_dict(), model_path)
        np.save(train_loss_path, train_losses)
        np.save(val_loss_path, val_losses)

    def save_cv_model_and_loss(self, model, fold, train_losses, val_losses, save_model=True):
        if save_model:
            model_path = f"{self.folder_path}/{self.model_name}_{self.model_number}.pt"
            torch.save(model.state_dict(), model_path)
        train_loss_path = f"{self.folder_path}/{self.model_name}_{self.model_number}_fold-{str(fold)}_train_loss.npy"
        val_loss_path = f"{self.folder_path}/{self.model_name}_{self.model_number}_fold-{str(fold)}_val_loss.npy"
        np.save(train_loss_path, train_losses)
        np.save(val_loss_path, val_losses)
        
    def save_cv_model(self, model, fold):
        model_path = f"{self.folder_path}/{self.model_name}_{self.model_number}_fold-{fold}.pt"
        torch.save(model.state_dict(), model_path)

    def save_test_results(self, results):
        with open(f"{self.folder_path}/model_summary.txt", "a", encoding="utf-8") as f:
            f.write(results)
