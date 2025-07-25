import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from utils.device import get_device
from utils.save import save_best_model
from training.metrics import compute_metrics


def train_model(model, dataset, config, tokenizer):
    device = get_device()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=config["hyperparameters"]["batch_size"], shuffle=True, num_workers=config["hyperparameters"]["num_workers"])

    optimizer = AdamW(model.parameters(), lr=config["hyperparameters"]["learning_rate"])
    total_steps = len(dataloader) * config["hyperparameters"]["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["hyperparameters"]["warmup_steps"],
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_bleu = -1
    patience_counter = 0

    for epoch in range(config["hyperparameters"]["epochs"]):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Evaluate metrics
        val_metrics = compute_metrics(model, dataloader, tokenizer, device)
        bleu = val_metrics.get("bleu", 0)

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | BLEU: {bleu:.4f}")

        # Early stopping
        if bleu > best_bleu:
            best_bleu = bleu
            patience_counter = 0
            save_best_model(model, config["paths"]["save_dir"], config["paths"]["best_model"])
        else:
            patience_counter += 1
            if patience_counter >= config["hyperparameters"]["patience"]:
                print("Early stopping triggered.")
                break
