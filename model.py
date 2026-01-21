import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pathlib import Path
import matplotlib.pyplot as plt

# path
data_path = Path("data/")
images_path = data_path / "dataset-finger-count-0to5"
train_path = images_path / "train"
test_path = images_path / "test"

IMG_SIZE = 128
BATCH_SIZE = 32

# dataloaders
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(degrees=(-30, 30)),
    transforms.ToTensor()
])
custom_image_transform = transforms.Compose([
        transforms.Resize((128, 128))
])

torch.manual_seed(67)
train_data = datasets.ImageFolder(root=train_path, transform=transform)
test_data = datasets.ImageFolder(root=test_path, transform=transform)

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


# Embedding
PATCH_SIZE = 16
COLOR_CHANNELS = 1

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=COLOR_CHANNELS, patch_size=PATCH_SIZE, embedding_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0) # creates patches
        self.flatten = nn.Flatten(start_dim=2, end_dim=3) # because we are flattening only width and height

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0 # can be patched

        return self.flatten(self.patcher(x)).permute(0, 2, 1) # because model requires [batch_size, image_size, embedding_size]

# MSA and MLP
class MSABlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        attention_outputs, _ = self.multihead_attention(query=x, key=x, value=x, need_weights=False)
        return attention_outputs
    
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim=768, dim_feedforward=3072, mlp_dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=dim_feedforward),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=dim_feedforward, out_features=embedding_dim),
            nn.Dropout(p=mlp_dropout)
        )

    def forward(self, x):
        return self.mlp(self.layer_norm(x))
    
# Transformer encoder
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim=768, num_heads=12, dim_feedforward=3072, attn_dropout=0.0, mlp_dropout=0.1):
        super().__init__()
        self.msa_block = MSABlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, dim_feedforward=dim_feedforward, mlp_dropout=mlp_dropout)

    def forward(self, x):
        msa_x = self.msa_block(x) + x # + x --- residual connections (adding input to the output to prevent gradient from getting too small)
        return self.mlp_block(msa_x) + msa_x
    

# MODEL
class ViT(nn.Module):
    def __init__(self,
                 img_size=IMG_SIZE,
                 in_channels=COLOR_CHANNELS,
                 patch_size=PATCH_SIZE,
                 num_transformer_layers:int=12,
                 embedding_dim=768,
                 dim_feedforward=3072, 
                 num_heads=12,
                 attn_dropout=0.0,
                 mlp_dropout=0.1,
                 embedding_dropout=0.1,
                 num_classes=6):
        super().__init__()

        assert img_size % patch_size == 0 # or else can't create equal patches and everything breaks

        self.num_of_patches = int((IMG_SIZE * IMG_SIZE) / PATCH_SIZE**2)


        # embedding and learnable class token + positional embedding
        self.class_embedding = nn.Parameter(torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_of_patches+1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)

        # Transformer Encoder created with blocks
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           dim_feedforward=dim_feedforward,
                                                                           attn_dropout=attn_dropout,
                                                                           mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1) # to match batch_size cause we don't know it on __init__()

        # embedding
        x = torch.cat((class_token, self.patch_embedding(x)), dim=1) # patch + class
        x = self.embedding_dropout(self.position_embedding + x)

        # transformer
        x = self.transformer_encoder(x)

        # classify
        return self.classifier(x[:, 0]) # look at each class_token, which has index 0
    

# helper fns
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
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
def train_step(model: nn.Module, data_loader: torch.utils.data.DataLoader, optim: torch.optim, scheduler: torch.optim.lr_scheduler, loss_fn: nn.Module, acc_fn):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (x, y) in enumerate(data_loader):

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()

        train_acc += acc_fn(y, torch.argmax(y_pred, dim=1))

    return train_loss / len(data_loader.dataset), train_acc / len(data_loader.dataset)

def test_step(model: nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: nn.Module, acc_fn):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):
            test_pred = model(x)

            test_loss += loss_fn(test_pred, y).item()
            test_acc += acc_fn(y, torch.argmax(test_pred, dim=1)) # calls softmax inside
    
    return test_loss/len(data_loader.dataset), test_acc/len(data_loader.dataset)

def training(model: torch.nn.Module, train_data_loader: torch.utils.data.DataLoader, test_data_loader: torch.utils.data.DataLoader, optim: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, loss_fn: torch.nn.Module, epochs: int, acc_fn = accuracy_fn):
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in range(epochs):
        train_loss, train_acc = train_step(model=model, data_loader=train_data_loader, optim=optim, scheduler=scheduler, loss_fn=loss_fn, acc_fn=acc_fn)
        test_loss, test_acc = test_step(model=model, data_loader=test_data_loader, loss_fn=loss_fn, acc_fn=acc_fn)

        # save data
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results


# saving model
MODEL_PATH = Path("models")
MODEL_NAME = "finger_count_model"

def save_model(model: nn.Module, model_path: Path, model_name: str):
    model_path.mkdir(parents=True, exist_ok=True)

    model_name = model_name + ".pth"
    model_save = model_path / model_name
    torch.save(obj=model.state_dict(), f=model_save)


# training
def train_and_save():
    epochs = 15
    vit_model = ViT()
    loss_fn = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(params=vit_model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.3)
    warmup_steps = 5
    warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda step: min((step+1) / warmup_steps, 1.0))
    circle = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=5, eta_min=0.0, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[warmup, circle], milestones=[warmup_steps])

    torch.manual_seed(67)

    results = training(model=vit_model, train_data_loader=train_dataloader, test_data_loader=test_dataloader, optim=optim, scheduler=scheduler, loss_fn=loss_fn, epochs=epochs)
    plot_loss_curves(results)
    save_model(vit_model, MODEL_PATH, MODEL_NAME)


# TO CALL
def get_saved_model() -> nn.Module:
    loaded_model = ViT()
    state_dict_path = MODEL_PATH / MODEL_NAME+".pth"

    if 'model_state_dict' in state_dict_path or 'state_dict' in state_dict_path:
        loaded_model.load_state_dict(torch.load(f=state_dict_path))
    else:
        print("First have to train model")

        train_and_save()
        loaded_model.load_state_dict(torch.load(f=state_dict_path))
    
    return loaded_model

def get_fingers(model: nn.Module, img, image_transform: transforms.Compose):
    img_transformed = image_transform(img).type(torch.float32)
    img_transformed /= 255

    model.eval()
    with torch.inference_mode():
        pred = model(img_transformed.unsqueeze(dim=0))

    pred_probs = torch.softmax(pred, dim=1)
    return int(torch.argmax(pred_probs, dim=1))