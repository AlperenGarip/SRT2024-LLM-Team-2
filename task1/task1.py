# token name: sri_llm_task, token value: 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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face API token from environment variable
hf_token = os.getenv('HF_HOME')

# Define model name
model_name = "SamLowe/roberta-base-go_emotions"

def load_model_and_tokenizer():
    # Load the model and tokenizer from Hugging Face Model Hub
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=hf_token)
    return tokenizer, model

def classify_emotions(text, tokenizer, model):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get logits and convert to probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the emotion labels
    emotion_labels = tokenizer.convert_ids_to_tokens(range(probabilities.size(-1)))
    
    # Filter emotions with probability greater than 5%
    threshold = 0.05 
    emotion_probs = probabilities[0].tolist()
    significant_emotions = [(emotion_labels[i], prob) for i, prob in enumerate(emotion_probs) if prob > threshold]
    
    return significant_emotions

if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()
    
    text = input("Enter a text to classify emotions: ")
    emotions = classify_emotions(text, tokenizer, model)
    print("Emotions with more than 5% probability:")
    for emotion, prob in emotions:
        print(f"{emotion}: {prob * 100:.2f}%")
