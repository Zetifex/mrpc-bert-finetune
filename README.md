# BERT Paraphrase Detector

This project fine-tunes a BERT model (`bert-base-uncased`) on the MRPC (Microsoft Research Paraphrase Corpus) dataset from the GLUE benchmark. It is designed to classify whether two sentences are semantically equivalent (paraphrases) or not.

## Features
- **Efficient Fine-Tuning**: Uses gradient accumulation and mixed-precision training (FP16) to allow training on GPUs with limited VRAM (e.g., 6GB).
- **Interactive Testing**: Includes a script to interactively test the model with your own sentence pairs.

## Files
- `transformer.py`: The main training script. It loads the MRPC dataset, fine-tunes the BERT model, and saves the trained weights.
- `test_model.py`: An interactive command-line tool to load the trained model and test it against custom sentences.

## Usage

1. **Train the Model**:
   ```bash
   python transformer.py
   ```
   *This will train the model and save it to a local folder named `./my_mrpc_model`.*

2. **Test the Model**:
   ```bash
   python test_model.py
   ```
   *You can type in sentences to see if the model detects them as paraphrases.*

## Known Limitations (The Lexical Overlap Trap)
Because the MRPC dataset is highly biased towards word overlap, the model may mistakenly classify sentences with the same words but different structures (e.g., "The chef cooked for the guests" vs "The guests cooked for the chef") as paraphrases. This is a common phenomenon in NLP!
