# Paraphrase Detection with BERT and QQP

This repository contains a robust paraphrase detection model fine-tuned on the **Quora Question Pairs (QQP)** dataset using a `bert-base-uncased` transformer. It identifies whether two sentences hold the exact same semantic meaning, regardless of the vocabulary used.

## 🚀 The Development Journey

This project evolved significantly during development to overcome common NLP pitfalls and hardware limitations:

### 1. Escaping the "Lexical Overlap Trap" (MRPC -> QQP)
We originally trained the model on the MRPC dataset. However, MRPC is heavily biased toward word overlap. Our model learned a "lazy" heuristic: *if the words are the same, it must be a paraphrase*. It failed on adversarial sentences like:
- *"The chef cooked for the guests."* vs *"The guests cooked for the chef."*

To fix this, we migrated to the **QQP dataset** (using a 50,000 example subset), which contains a massive variety of adversarial examples, tricky reversals, and true paraphrases. 

### 2. Preventing Hardware Overheating (Accelerate & FP16)
Full fine-tuning of 110-million parameters caused severe laptop overheating. We resolved this by integrating **Hugging Face Accelerate** and enforcing **FP16 Mixed Precision**. 
* FP16 utilizes Tensor Cores to drastically reduce memory bandwidth and power consumption.
* *(Note: We temporarily experimented with PEFT/LoRA, but due to known serialization bugs with Sequence Classification heads in the PEFT library, we reverted to full fine-tuning. FP16 alone was enough to keep the hardware cool and fast!)*

## 📂 Files
- `transformer.py`: The robust training script. It handles FP16 acceleration, dataset downloading, learning rate warmup, and model saving.
- `test_model.py`: An interactive testing tool to load the model and interrogate it with your own tricky sentences.

## 🧪 Complicated Tests to Try
Once trained, run `python test_model.py` and try these adversarial tests that break weaker models:

**Test 1: The "Vocabulary Swap" (Should output 🟢 Paraphrase)**
* Sentence 1: *"What are the most effective methods to shed belly fat quickly?"*
* Sentence 2: *"How can I rapidly lose weight around my midsection?"*
* *Why it's hard:* Almost zero word overlap, but identical meaning.

**Test 2: The "Tricky Reversal" (Should output 🔴 DIFFERENT)**
* Sentence 1: *"How do I start learning Python if I already know Java?"*
* Sentence 2: *"How do I start learning Java if I already know Python?"*
* *Why it's hard:* 100% word overlap, but the core intent is completely reversed.

## ⚙️ Usage
**Train the model (10-15 minutes):**
```bash
python transformer.py
```

**Interact and test:**
```bash
python test_model.py
```
