import os, time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TxtDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_len=128):
        self.texts=texts; self.labels=labels; self.tokenizer=tokenizer; self.max_len=max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        t=self.texts[idx]
        enc=self.tokenizer(t, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item={k:enc[k].squeeze(0) for k in enc}
        if self.labels is not None:
            item['labels']=torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def run_text_pipeline(data_path, competition_id, output_path, model_name="distilbert-base-uncased", epochs=1, batch_size=16):
    base=Path(data_path)/competition_id
    if not base.exists(): base = Path(data_path)
    print("Text pipeline base:", base)
    train = base/"train.csv"; sample = base/"sample_submission.csv"
    if not train.exists():
        print("No train.csv; writing dummy sample_submission")
        if sample.exists(): pd.read_csv(sample).assign(**{c:0 for c in pd.read_csv(sample).columns[1:]}).to_csv(output_path,index=False)
        return
    df=pd.read_csv(train)
    # naive column detection: text column contains 'text' or 'excerpt' or 'sentence'
    text_col = next((c for c in df.columns if 'text' in c.lower() or 'sentence' in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if c not in [text_col]), None)
    if label_col is None:
        print("No label col found; writing zeros")
        pd.read_csv(sample).to_csv(output_path,index=False); return
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = TxtDataset(texts, labels, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(labels))).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(loader, desc=f"train-epoch{epoch+1}")
        for batch in pbar:
            optim.zero_grad()
            input_ids=batch['input_ids'].to(device); attn=batch['attention_mask'].to(device); labelsb=batch['labels'].to(device)
            outputs=model(input_ids, attention_mask=attn, labels=labelsb)
            loss=outputs.loss; loss.backward(); optim.step()
            pbar.set_postfix({'loss':float(loss)})
    # inference on sample_submission inputs if available
    sample_df = pd.read_csv(sample) if sample.exists() else None
    if sample_df is None:
        print("No sample_submission to create predictions; writing nothing")
        return
    # For many text competitions sample_submission contains ids and blank pred col; need to map
    texts_to_pred = sample_df.iloc[:,1].astype(str).tolist() if sample_df.shape[1]>1 else [""]*len(sample_df)
    pred_ds = TxtDataset(texts_to_pred, labels=None, tokenizer=tokenizer)
    pred_loader = DataLoader(pred_ds, batch_size=batch_size)
    preds=[]
    model.eval()
    with torch.no_grad():
        for b in pred_loader:
            input_ids=b['input_ids'].to(device); attn=b['attention_mask'].to(device)
            logits = model(input_ids, attention_mask=attn).logits
            labs = logits.argmax(dim=1).cpu().numpy().tolist()
            preds.extend(labs)
    # map preds into sample_submission shape
    out = sample_df.copy()
    target_col = out.columns[1] if out.shape[1]>1 else out.columns[-1]
    out[target_col] = preds
    out.to_csv(output_path, index=False)
    print("Saved text submission to", output_path)
