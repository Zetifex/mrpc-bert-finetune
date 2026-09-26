import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import os

def test_model():
    model_path = "./my_qqp_model"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find the model at {model_path}.")
        print("Please run transformer.py first so it can train and save the PAWS model!")
        return

    print("Loading the trained model...")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # With LoRA, we load the base model first, then layer the small trained adapter on top!
    base_model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    model = PeftModel.from_pretrained(base_model, model_path)
    
    print("\n--- Model Loaded Successfully ---")
    print("Type 'quit' to exit.")
    
    while True:
        print("-" * 50)
        sentence1 = input("Sentence 1: ")
        if sentence1.lower() == 'quit':
            break
            
        sentence2 = input("Sentence 2: ")
        if sentence2.lower() == 'quit':
            break
            
        # 1. Tokenize the input sentences
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", truncation=True)
        
        # 2. Pass them through the model
        with torch.no_grad():
            outputs = model(**inputs)
            
        # 3. Get the prediction (0 or 1)
        logits = outputs.logits
        prediction_id = torch.argmax(logits, dim=-1).item()
        
        # 4. In the MRPC dataset: 1 = Paraphrase, 0 = Not a Paraphrase
        print("\n=> RESULT:")
        if prediction_id == 1:
            print("🟢 The model thinks these sentences MEAN THE SAME THING (Paraphrase).")
        else:
            print("🔴 The model thinks these sentences are DIFFERENT.")

if __name__ == "__main__":
    test_model()
