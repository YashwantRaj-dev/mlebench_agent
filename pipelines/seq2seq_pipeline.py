import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

def run_seq2seq_pipeline(data_path, competition_id, output_path, model_name="t5-small", epochs=1, batch_size=8):
    base = Path(data_path)/competition_id
    if not base.exists(): base = Path(data_path)
    train = base/"train.csv"; sample = base/"sample_submission.csv"
    if not train.exists():
        print("train.csv missing; writing empty sample_submission if exists")
        if sample.exists(): pd.read_csv(sample).to_csv(output_path,index=False)
        return
    df = pd.read_csv(train)
    # assume columns 'before' and 'after' or similar
    text_col = next((c for c in df.columns if 'before' in c.lower() or 'input' in c.lower() or 'text' in c.lower()), df.columns[0])
    target_col = next((c for c in df.columns if c!=text_col), None)
    if target_col is None:
        print("No target column found — skipping")
        if sample.exists(): pd.read_csv(sample).to_csv(output_path,index=False)
        return
    tokenizer = T5TokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # tiny training: prepare input_ids
    inputs = tokenizer(list(df[text_col].astype(str)), truncation=True, padding=True, return_tensors='pt')
    targets = tokenizer(list(df[target_col].astype(str)), truncation=True, padding=True, return_tensors='pt')
    dataset = torch.utils.data.TensorDataset(inputs['input_ids'], inputs['attention_mask'], targets['input_ids'])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"t5-train-{epoch+1}")
        for input_ids, attn, target_ids in pbar:
            input_ids=input_ids.to(device); attn=attn.to(device); target_ids=target_ids.to(device)
            outputs=model(input_ids=input_ids, attention_mask=attn, labels=target_ids)
            loss=outputs.loss
            loss.backward()
            optim.step(); optim.zero_grad()
            pbar.set_postfix({"loss":float(loss)})
    # inference: use sample_submission id or available input column
    if sample.exists():
        sample_df = pd.read_csv(sample)
        # try to find input column; else map ids to blank
        inputs_to_pred = list(sample_df.iloc[:,1].astype(str)) if sample_df.shape[1]>1 else [""]*len(sample_df)
        outputs=[]
        model.eval()
        for chunk_start in range(0,len(inputs_to_pred),16):
            chunk = inputs_to_pred[chunk_start:chunk_start+16]
            enc = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True).to(device)
            generated = model.generate(**enc, max_length=128)
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(decoded)
        # write back into sample_submission's second column if exists
        out = sample_df.copy()
        if out.shape[1]>1:
            out.iloc[:,1] = outputs[:len(out)]
        out.to_csv(output_path, index=False)
        print("Saved seq2seq submission to", output_path)
    else:
        print("No sample_submission found; nothing saved")
