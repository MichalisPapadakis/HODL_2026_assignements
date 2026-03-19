import os
import copy
import random
import glob
import zipfile
import urllib.request
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights
import matplotlib.pyplot as plt
from PIL import Image

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


#########
## STEP 0: Global Configuration (minimal, no CLI args)
#########
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_FREEZE = 5
EPOCHS_FULL = 5
LR_FREEZE = 1e-3
LR_FULL = 1e-4
WEIGHT_DECAY = 1e-4

DATA_ROOT = "data_scratch"
ZIP_PATH = os.path.join(DATA_ROOT, "kagglecatsanddogs_5340.zip")
PET_IMAGES_DIR = os.path.join(DATA_ROOT, "PetImages")
DOWNLOAD_URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

SAVE_PATH = "./checkpoints/vgg_finetuned.pt"
INFER_IMAGE = None


#########
## STEP A: Core Utilities
## Reproducibility and shared image transforms.
#########


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def denormalize(img_tensor: torch.Tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1)


#########
## STEP B: Data Download, Cleaning, and Loading
## Dataset integrity checks, extraction, split, and dataloaders.
#########


def _check_image(fn: str) -> bool:
    try:
        im = Image.open(fn)
        im.verify()
        return True
    except Exception:
        return False


def _cleanup_corrupt_images(path_glob: str):
    for fn in glob.glob(path_glob):
        if not _check_image(fn):
            print(f"Removing corrupt image: {fn}")
            os.remove(fn)


def ensure_petimages_dataset():
    os.makedirs(DATA_ROOT, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print("Downloading Cats vs Dogs dataset...")
        urllib.request.urlretrieve(DOWNLOAD_URL, ZIP_PATH)

    cat_dir = os.path.join(PET_IMAGES_DIR, "Cat")
    dog_dir = os.path.join(PET_IMAGES_DIR, "Dog")
    if not (os.path.isdir(cat_dir) and os.path.isdir(dog_dir)):
        print("Extracting dataset...")
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(DATA_ROOT)

    print("Checking for corrupt images...")
    _cleanup_corrupt_images(os.path.join(PET_IMAGES_DIR, "Cat", "*.jpg"))
    _cleanup_corrupt_images(os.path.join(PET_IMAGES_DIR, "Dog", "*.jpg"))


def load_data(batch_size: int, img_size: int):
    tfm = get_transform(img_size)
    full_ds = datasets.ImageFolder(PET_IMAGES_DIR, transform=tfm)

    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return full_ds, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


#########
## STEP C: Visualization and Feature Inspection
## Quick sample preview plus feature-map visualization.
#########


def show_example_image(dataset: datasets.ImageFolder, idx: int | None = None):
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    x, y = dataset[idx]
    x = denormalize(x)

    plt.figure(figsize=(4, 4))
    plt.imshow(x.permute(1, 2, 0).cpu().numpy())
    plt.title(f"Example image | class: {dataset.classes[y]}")
    plt.axis("off")
    plt.tight_layout()
    # plt.show()

    return idx


#########
## STEP D: Model Construction
## Build pre-trained backbone and adapt classification head.
#########


def build_vgg_model(num_classes: int):
    weights = VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    return model


def inspect_pretrained_features(model: nn.Module, sample_batch: torch.Tensor, device: torch.device):
    """
    Visualize VGG feature maps for one input image.

    - Rows: selected layers (4 layers)
    - Columns: first feature channels (5 channels per layer)
    - The input is the same sample image used earlier in STEP 1 (if you pass that sample).
    """
    model.eval()
    sample = sample_batch[:1].to(device)

    # 4 indicative layers from VGG feature extractor
    layer_ids = [0, 5, 10, 17]
    layer_desc = {
        0:  "Early edges / color blobs",
        5:  "Simple textures / corners",
        10: "Mid-level patterns",
        17: "Higher-level shapes/parts",
    }

    n_features_to_show = 5
    feature_maps: Dict[int, torch.Tensor] = {}

    hooks = []
    for li in layer_ids:
        def _make_hook(layer_idx):
            def _hook(_, __, output):
                feature_maps[layer_idx] = output.detach().cpu()  # [B, C, H, W]
            return _hook
        hooks.append(model.features[li].register_forward_hook(_make_hook(li)))

    with torch.no_grad():
        _ = model(sample)

    for h in hooks:
        h.remove()

    # Grid: 4 rows (layers) x 5 cols (channels)
    fig, axes = plt.subplots(
        nrows=len(layer_ids),
        ncols=n_features_to_show,
        figsize=(2.4 * n_features_to_show, 2.3 * len(layer_ids)),
        squeeze=False
    )

    for r, li in enumerate(layer_ids):
        fmap = feature_maps[li][0]  # first sample -> [C, H, W]
        c_max = min(n_features_to_show, fmap.shape[0])

        for c in range(n_features_to_show):
            ax = axes[r, c]
            if c < c_max:
                ax.imshow(fmap[c].numpy(), cmap="viridis")
                ax.set_title(f"ch {c}", fontsize=9)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=8)
            ax.axis("off")

            if c == 0:
                ax.set_ylabel(f"features[{li}]\n{layer_desc[li]}", fontsize=9)

    fig.suptitle("VGG pre-trained feature maps (4 layers × 5 channels)", fontsize=12)
    fig.text(
        0.5, 0.01,
        "Description: each row is a deeper layer; each column is one channel (feature detector). "
        "Lower layers capture edges/textures, deeper layers capture more abstract shapes.",
        ha="center", fontsize=9
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


#########
## STEP E: Training and Evaluation
## Epoch loops, validation, and best-checkpoint selection.
#########


def train_one_epoch(model, loader, criterion, optimizer, device, show_progress: bool = True, desc: str = "Train"):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False)

    for x, y in iterator:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == y).sum().item()
        total += x.size(0)

        if show_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{running_correct / total:.4f}")

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, show_progress: bool = False, desc: str = "Val"):
    model.eval()
    running_loss, running_correct, total = 0.0, 0, 0

    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(loader, desc=desc, leave=False)

    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == y).sum().item()
        total += x.size(0)

        if show_progress and tqdm is not None:
            iterator.set_postfix(loss=f"{running_loss / total:.4f}", acc=f"{running_correct / total:.4f}")

    return running_loss / total, running_correct / total


def fit(model, train_loader, val_loader, device, epochs, lr, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    best_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            show_progress=True,
            desc=f"Train {epoch}/{epochs}"
        )
        va_loss, va_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
            show_progress=True,
            desc=f"Val {epoch}/{epochs}"
        )

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    return model, best_val_acc


#########
## STEP F: Inference Utility
## Single-image prediction with confidence score.
#########


def run_inference(model, image_path: str, class_names: List[str], device: torch.device, img_size: int = 224):
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    pred_class = class_names[pred_idx]
    conf = probs[pred_idx].item()

    print(f"Inference -> class: {pred_class}, confidence: {conf:.4f}")
    return pred_class, conf


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #########
    ## STEP 1: Load and Inspect Data
    #########
    ensure_petimages_dataset()
    full_ds, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = load_data(
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Classes ({len(full_ds.classes)}): {full_ds.classes}")
    sample_idx = show_example_image(full_ds)

    #########
    ## STEP 2: Inspect pre-trained features (VGG)
    #########
    model = build_vgg_model(num_classes=len(full_ds.classes)).to(device)
    sample_x, _ = full_ds[sample_idx]
    sample_batch = sample_x.unsqueeze(0)
    inspect_pretrained_features(model, sample_batch, device)

    #########
    ## STEP 3: Train (freeze weights)
    #########
    for p in model.features.parameters():
        p.requires_grad = False

    # classifier remains trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    model, best_acc_freeze = fit(
        model, train_loader, val_loader, device,
        epochs=EPOCHS_FREEZE,
        lr=LR_FREEZE,
        weight_decay=WEIGHT_DECAY
    )
    print(f"Best val acc (frozen backbone): {best_acc_freeze:.4f}")

    #########
    ## STEP 4: Train (whole network)
    #########
    for p in model.parameters():
        p.requires_grad = True

    model, best_acc_full = fit(
        model, train_loader, val_loader, device,
        epochs=EPOCHS_FULL,
        lr=LR_FULL,
        weight_decay=WEIGHT_DECAY
    )
    print(f"Best val acc (full fine-tune): {best_acc_full:.4f}")

    #########
    ## STEP 5: Save model
    #########
    save_dir = os.path.dirname(SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": full_ds.class_to_idx,
        "idx_to_class": {v: k for k, v in full_ds.class_to_idx.items()},
        "num_classes": len(full_ds.classes),
        "img_size": IMG_SIZE,
    }
    torch.save(ckpt, SAVE_PATH)
    print(f"Model saved to: {SAVE_PATH}")

    #########
    ## STEP 6: Use model for inference
    #########
    if INFER_IMAGE:
        idx_to_class = ckpt["idx_to_class"]
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        run_inference(model, INFER_IMAGE, class_names, device, img_size=IMG_SIZE)
    else:
        # quick test-set evaluation if no single image is provided
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()