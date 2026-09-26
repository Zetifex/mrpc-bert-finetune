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
from accelerate import Accelerator

def main():
    # 1. Initialize Accelerator
    # Accelerate handles mixed precision and gradient accumulation automatically
    accelerator = Accelerator(gradient_accumulation_steps=4)

    # 2. Load Data and Tokenizer
    # Switched to PAWS (Paraphrase Adversaries from Word Scrambling)
    # This dataset specifically fixes the "Lexical Overlap" trap!
    raw_datasets = load_dataset("paws", "labeled_final")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # PAWS uses 'id' instead of 'idx'
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "id"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # 3. Dataloaders
    num_workers = min(4, os.cpu_count() or 1)
    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        shuffle=True, 
        batch_size=8, # Slightly larger batch size since Accelerate optimizes memory well
        collate_fn=data_collator,
        pin_memory=True,          
        num_workers=num_workers   
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        batch_size=8, 
        collate_fn=data_collator,
        pin_memory=True,
        num_workers=num_workers
    )

    # 4. Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    # 5. Set up Optimizer 
    optimizer = AdamW(model.parameters(), lr=5e-5)

    num_epochs = 3
    num_training_steps = num_epochs * (len(train_dataloader) // accelerator.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # 6. Prepare everything with Accelerate
    # This magically moves models/tensors to GPUs and prepares them for training
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # 7. The Training Loop (Notice how much cleaner this is!)
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_local_main_process)
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                # Accelerate handles mixed-precision scaling and backward passes
                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                
            # Only update progress bar when we actually take an optimizer step
            if accelerator.sync_gradients:
                progress_bar.update(1)

    # 8. The Evaluation Loop
    metric = evaluate.load("accuracy")
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Accelerate can gather predictions across multiple GPUs if you ever scale up
        predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
        metric.add_batch(predictions=predictions, references=references)

    results = metric.compute()
    accelerator.print(f"Evaluation Results: {results}")

    # 9. Save the trained model and tokenizer
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        print("Saving the model for testing...")
        unwrapped_model.save_pretrained("./my_paws_model")
        tokenizer.save_pretrained("./my_paws_model")
        print("Model successfully saved to ./my_paws_model!")

if __name__ == '__main__':
    main()