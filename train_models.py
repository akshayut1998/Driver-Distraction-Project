# train_models.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator
from custom_cnn_model import CustomCNN
from resnet_model import ResnetCNN
from vision_transformer_model import VisionTransformerModel
from efficientnet_model import EfficientNetModel
from Seed_LR import set_random_seeds,find_lr

def cnntrain(train_loader,val_loader,combined_loader,len_data,num_filters="Pretrained-Model", num_epochs=5, train_only=False, custom="Custom",learning_rate=0.0005):
    set_random_seeds()
    model_best_val_losses = []
    model_best_epochs = []
    model_objects = []

    if custom != "Custom":
        num_filters = "D"
    for i in num_filters:

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        # Create an instance of the custom CNN
        if custom == "Custom":
            model = CustomCNN(len_data, i)
        elif custom == "Resnet":
            model = ResnetCNN(len_data)
        elif custom == "VIT":
            model = VisionTransformerModel(len_data)
        else:
            model = EfficientNetModel(len_data)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if not train_only:
            # Loss function and optimizer
            try:
                learning_rate = find_lr(train_loader, model, criterion, optimizer, device)
                print(f"Selected learning rate from LR range test: {learning_rate}")
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            except:
                pass

        # Training loop
        best_epochs = 1
        for epoch in range(1,num_epochs+1):
            model.train()

            loader_to_use = combined_loader if train_only else train_loader

            # Use tqdm to create a progress bar for the train_loader
            number_of_mini_batches=0
            with tqdm(loader_to_use, desc=f'Epoch {epoch}/{num_epochs}', leave=False) as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                    train_losses.append(loss.item())
                    number_of_mini_batches=number_of_mini_batches+1
                    
            if not train_only:

                model.eval()
                val_loss = 0.0
                total=0.0
                correct=0.0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        val_loss += criterion(outputs, labels).item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100 * correct / total
                print(f"Validation Accuracy: {accuracy:.2f}%")

                val_loss /= len(val_loader)

                val_losses.append(val_loss)
                print(f'Validation Loss: {val_loss:.2f}')

                # Check for early stopping
                if val_loss < best_val_loss or (epoch <= 2):
                    best_model_weights = model.state_dict()
                    best_val_loss = val_loss
                    best_epochs = epoch
                else:
                    print("Early stopping triggered. Restoring best model weights.\n")
                    model.load_state_dict(best_model_weights)
                    break

        if not train_only:
            model.load_state_dict(best_model_weights)
            updates_range = range(1, len(train_losses) + 1)
            plt.figure(figsize=(20,12))
            plt.plot(updates_range, train_losses, label='Training Loss', linewidth=0.5)
            plt.plot(np.linspace(number_of_mini_batches,number_of_mini_batches*len(val_losses)\
                ,len(val_losses)),val_losses , label='Validation Loss', linewidth=1.5)
            for epochs in range(1, epoch + 1):
                if epochs==1:
                    plt.axvline(x=epochs * number_of_mini_batches, color='gray', linestyle='--', linewidth=1, label=f'Epochs')
                elif epochs==best_epochs:
                    plt.axvline(x=epochs * number_of_mini_batches, color='green', linestyle='-', linewidth=2, label=f'Optimals epochs is {best_epochs}')
                elif epochs==best_epochs+1:
                    plt.axvline(x=epochs * number_of_mini_batches, color='red', linestyle='-', linewidth=2, label=f'Early stopping initiated at epoch {best_epochs+1}')
                else:
                    plt.axvline(x=epochs * number_of_mini_batches, color='gray', linestyle='--', linewidth=1)
            plt.yscale('log')
            plt.xlabel('Number of Weight Updates', fontsize=20)
            plt.ylabel('Loss (Log Scale)', fontsize=20)
            plt.tick_params(axis='both', which='both', labelsize=20)
            plt.legend(loc='best', fontsize=20)
            plt.show()

            model_best_val_losses.append(best_val_loss)
            model_best_epochs.append(best_epochs)
            model_objects.append(model)
    if not train_only:
        return model_best_val_losses, model_best_epochs, model_objects , learning_rate
    else:
        return model