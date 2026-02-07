import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import splitfolders
import shutil
from pathlib import Path
# import os

import matplotlib.pyplot as plt

# split dataset
def split_dataset(all_images_path: str, split_path: Path, train_path: Path, val_path: Path):
    splitfolders.ratio(all_images_path, output=split_path, seed=67, ratio=(0.8, 0.2), group_prefix=None)
    for i in range (6):
        (train_path/str(i)).mkdir()
        (val_path/str(i)).mkdir()

    for img in train_path.iterdir():
        if img.is_dir():
            continue
        full_path = str(img)
        shutil.move(full_path, str(train_path) + "/" + full_path[len(full_path)-5]) # split to folders

    for img in val_path.iterdir():
        if img.is_dir():
            continue
        full_path = str(img)
        shutil.move(full_path, str(val_path) + "/" + full_path[len(full_path)-5]) # split to folders

# path
all_images_path = "data/dataset-hands-black-white"
split_path = Path("data/dataset-hands-black-white-split")
train_path = split_path / "train"
val_path = split_path / "val"

if not split_path.is_dir(): # split data
    split_dataset(all_images_path, split_path, train_path, val_path)

# settings
IMG_SIZE = 128
BATCH_SIZE = 20
EMBEDDING_DIM = 64
COLOR_CHANNELS = 1
NUM_WORKERS = 0 # os.cpu_count()

# dataloaders
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.RandomAffine(degrees=0, shear=25),
    transforms.RandomErasing(p=0.5),
    transforms.Grayscale(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
custom_image_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

torch.manual_seed(67)
train_data = datasets.ImageFolder(root=train_path, transform=transform)
val_data = datasets.ImageFolder(root=val_path, transform=transform)

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


# Embedding
MULT = 8*8
class CNN(nn.Module):
    def __init__(self, input_shape=COLOR_CHANNELS, hidden_units=EMBEDDING_DIM, output_shape=6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d( # analyze each square (chunk, batch) of data to find connections, change dimensions to do autograd better
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=2 # step
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3, 
                stride=2
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=1, stride=2) # compresses the matrix and leaves only the most important
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*MULT, out_features=hidden_units*MULT),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(in_features=hidden_units*MULT, out_features=output_shape)
        )

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


# helper fns
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred))
    return acc

def plot_loss_curves(results):
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


# training fns
def train_step(model: nn.Module, data_loader: torch.utils.data.DataLoader, optim: torch.optim, loss_fn: nn.Module, scheduler: torch.optim.lr_scheduler = None, acc_fn = accuracy_fn, device="cpu"):
    train_loss, train_acc = 0, 0
    model.train()
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        temp_probs = torch.softmax(y_pred, dim=1)
        entropy = torch.mean(temp_probs * torch.log(temp_probs + 1e-10))
        loss = loss + 0.1 * (-entropy)  # Penalize low entropy (collapsed probs)

        train_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_acc += acc_fn(y, torch.argmax(torch.softmax(y_pred, dim=1), dim=1))

    if scheduler:
        scheduler.step()

    return train_loss / (batch+1), train_acc / (batch+1)

def test_step(model: nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: nn.Module, acc_fn=accuracy_fn, device="cpu"):
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            test_pred = model(x)

            test_loss += loss_fn(test_pred, y).item()
            test_acc += acc_fn(y, torch.argmax(torch.softmax(test_pred, dim=1), dim=1))

    return test_loss / (batch+1), test_acc / (batch+1)

def training(model: torch.nn.Module, train_data_loader: torch.utils.data.DataLoader, test_data_loader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, loss_fn: torch.nn.Module, epochs: int, scheduler: torch.optim.lr_scheduler = None, acc_fn = accuracy_fn, device="cpu"):
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model, data_loader=train_data_loader, optim=optim, scheduler=scheduler, loss_fn=loss_fn, acc_fn=acc_fn, device=device)
        test_loss, test_acc = test_step(model=model, data_loader=test_data_loader, loss_fn=loss_fn, acc_fn=acc_fn, device=device)

        print(f"Epoch: {epoch}")
        print(f"train_loss: {train_loss*100:.3f}% | train_acc: {train_acc*100:.3f}%")
        print(f"test_loss: {test_loss*100:.3f}% | test_acc: {test_acc*100:.3f}%")
        print("")

        # save data
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results


# saving model
MODEL_PATH = Path("models")
MODEL_NAME = "kid_named_finger_2.4"
device = "cuda" if torch.cuda.is_available() else "cpu"

def save_model(model: nn.Module):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    model_save = MODEL_PATH / (MODEL_NAME + '.pth')
    torch.save(obj=model.state_dict(), f=model_save)


# training
def train_and_save():
    epochs = 7
    cnn_model = CNN().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optim = torch.optim.AdamW(params=cnn_model.parameters(), lr=0.001, weight_decay=0.2, betas=[0.9, 0.999])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim,T_0=10,T_mult=2,eta_min=1e-6, last_epoch=-1)

    torch.manual_seed(67)
    results = training(model=cnn_model, train_data_loader=train_dataloader, test_data_loader=val_dataloader, optim=optim, scheduler=scheduler, loss_fn=loss_fn, epochs=epochs, device=device)

    plot_loss_curves(results)
    save_model(cnn_model)


# TO CALL
def get_saved_model() -> nn.Module:
    loaded_model = CNN()
    state_dict_path = MODEL_PATH / (MODEL_NAME + ".pth")

    if state_dict_path.is_file():
        loaded_model.load_state_dict(torch.load(f=state_dict_path, map_location=torch.device(device)))
    else:
        print("First have to train model")
        train_and_save()
        loaded_model.load_state_dict(torch.load(f=state_dict_path, map_location=torch.device(device)))

    return loaded_model

import numpy as np
def get_fingers(model: nn.Module, img):
    torch_img = torch.from_numpy(img).permute(2, 0, 1).type(torch.float32)
    img_transformed = custom_image_transform(torch_img)

    model.eval()
    with torch.inference_mode():
        pred = model(img_transformed.unsqueeze(dim=0))

    pred_probs = torch.softmax(pred, dim=1)
    return int(torch.argmax(pred_probs, dim=1))


# train_and_save()