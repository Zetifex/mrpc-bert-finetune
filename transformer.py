import torch
import evaluate
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    DataCollatorWithPadding, 
    AutoModelForSequenceClassification, 
    get_scheduler
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

def main():
    # 1. Load data and tokenizer
    raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # 2. VRAM-Optimized Dataloaders
    num_workers = min(4, os.cpu_count() or 1)
    # Physical batch size reduced to 4 to prevent Out Of Memory (OOM) errors on 6GB VRAM
    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        shuffle=True, 
        batch_size=4, 
        collate_fn=data_collator,
        pin_memory=True,          
        num_workers=num_workers   
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        batch_size=4, 
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=num_workers
    )

    # 3. Initialize model and Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model.to(device)

    # PyTorch 2.0 Optimization
    # torch.compile() uses Triton, which is not natively supported on Windows.
    if hasattr(torch, "compile") and torch.cuda.is_available() and os.name != "nt":
        model = torch.compile(model)

    # 4. Set up Optimizer and Gradient Accumulation
    is_cuda = torch.cuda.is_available()
    optimizer = AdamW(model.parameters(), lr=5e-5, fused=is_cuda)

    # Gradient Accumulation: Simulates a larger batch size without exceeding VRAM
    # Effective batch size = Physical batch size (4) * Accumulation steps (4) = 16
    accumulation_steps = 4 

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader) // accumulation_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # 5. Initialize FP16 GradScaler (Leveraging Turing Tensor Cores)
    scaler = torch.amp.GradScaler('cuda') if is_cuda else None

    # 6. The Training Loop with Accumulation
    progress_bar = tqdm(range(num_training_steps))
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', enabled=is_cuda):
                outputs = model(**batch)
                # Normalize the loss to account for accumulation
                loss = outputs.loss / accumulation_steps
                
            if scaler is not None:
                scaler.scale(loss).backward()
                
                # Only update weights after accumulating enough gradients
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True) 
                    progress_bar.update(1)
            else:
                loss.backward()
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    progress_bar.update(1)

    # 7. The Evaluation Loop
    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=is_cuda):
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    results = metric.compute()
    print(results)

    # 8. Save the trained model and tokenizer
    print("Saving the model for testing...")
    model.save_pretrained("./my_mrpc_model")
    tokenizer.save_pretrained("./my_mrpc_model")
    print("Model successfully saved to ./my_mrpc_model!")

if __name__ == '__main__':
    main()