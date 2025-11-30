import os, zipfile
from pathlib import Path
import pandas as pd, numpy as np
from PIL import Image
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

def _ensure_unzipped(base:Path):
    for name in ("train","test"):
        d = base / name
        z = base / f"{name}.zip"
        if not d.exists() and z.exists():
            print(f"Extracting {z} -> {d}")
            with zipfile.ZipFile(z, 'r') as zf:
                zf.extractall(d)

class ImgDataset(Dataset):
    def __init__(self, df, root, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        self.is_test = is_test
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row.get("id") or row.get("image") or str(row.iloc[0])
        p = self.root / fname
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        if self.is_test: return img, fname
        return img, int(row["has_cactus"] if "has_cactus" in row else row.get("target",0))

def build_transforms(image_size=224):
    train = transforms.Compose([transforms.Resize((image_size,image_size)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(12), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    val = transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return train, val

def build_model(num_classes=2, device='cpu'):
    try:
        model = models.efficientnet_b0(pretrained=True)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f,num_classes))
    except Exception:
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def simple_train(model, loader, opt, crit, device, scaler=None):
    model.train()
    pbar = tqdm(loader, desc="train", leave=False)
    running = 0.0
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            out = model(imgs)
            loss = crit(out, labels)
        if scaler is not None:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        running += loss.item()
        pbar.set_postfix({"loss":f"{loss.item():.4f}"})
    return running/ max(1,len(loader))

def infer(model, loader, device):
    model.eval()
    res=[]
    pbar=tqdm(loader,desc="inference",leave=False)
    with torch.no_grad():
        for imgs, fnames in pbar:
            imgs=imgs.to(device)
            out=model(imgs)
            probs=torch.softmax(out,dim=1)[:,1].cpu().numpy()
            for f,p in zip(fnames,probs): res.append((f,float(p)))
    return res

def run_image_pipeline(data_path, competition_id, output_path, epochs=2, batch_size=32, image_size=224):
    base = Path(data_path) / competition_id
    if not base.exists(): base = Path(data_path)
    print("Image pipeline base:", base)
    _ensure_unzipped(base)
    train_csv=base/"train.csv"; sample=base/"sample_submission.csv"; train_dir=base/"train"; test_dir=base/"test"
    if not train_csv.exists(): raise FileNotFoundError(train_csv)
    if not train_dir.exists(): raise FileNotFoundError(f"{train_dir} missing; unzip train.zip")
    df_train=pd.read_csv(train_csv); df_test=pd.read_csv(sample)
    train_tf,val_tf=build_transforms(image_size)
    full_ds=ImgDataset(df_train, train_dir, transform=train_tf, is_test=False)
    total=len(full_ds); val_size=max(1,int(0.15*total)); train_size=total-val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    # val dataset wrapper to apply val_tf
    val_idx = val_ds.indices if hasattr(val_ds,"indices") else []
    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    val_dataset = ImgDataset(df_val, train_dir, transform=val_tf, is_test=False)
    train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=build_model(num_classes=2, device=device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        train_loss = simple_train(model, train_loader, opt, crit, device, scaler)
        print(f"train_loss: {train_loss:.4f}")
    # inference
    test_ds = ImgDataset(df_test, test_dir, transform=val_tf, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    preds = infer(model, test_loader, device)
    out = pd.DataFrame(preds, columns=["id","has_cactus"])
    # reorder like sample
    samp=pd.read_csv(sample)
    out = out.set_index("id").reindex(samp["id"]).reset_index()
    out.to_csv(output_path, index=False)
    print("Saved image submission to", output_path)
