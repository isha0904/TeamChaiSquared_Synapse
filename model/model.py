##This is the code from the Python notebook that was used to build and train the model. 



# Install necessary packages
!pip install transformers datasets sentencepiece sacrebleu torch accelerate evaluate

# Import required libraries
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# Load model and tokenizer
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load datasets for Bengali and Marathi
dataset_bn = load_dataset("ai4bharat/samanantar", "bn")
dataset_mr = load_dataset("ai4bharat/samanantar", "mr")

# Function to sample dataset
def sample_dataset(dataset, src_lang_code, tgt_lang_code):
    inputs = [dataset["train"][i]["src"] for i in range(0, len(dataset["train"]), 10000)]
    targets = [dataset["train"][i]["tgt"] for i in range(0, len(dataset["train"]), 10000)]
    lang_codes = [(src_lang_code, tgt_lang_code)] * len(inputs)
    return inputs, targets, lang_codes

# Sample data for Bengali and Marathi
inputs_bn, targets_bn, lang_codes_bn = sample_dataset(dataset_bn, "eng_Latn", "ben_Beng")
inputs_mr, targets_mr, lang_codes_mr = sample_dataset(dataset_mr, "eng_Latn", "mar_Deva")

# Combine datasets
inputs = inputs_bn + inputs_mr
targets = targets_bn + targets_mr
lang_codes = lang_codes_bn + lang_codes_mr

# Split into training and evaluation sets
train_inputs, eval_inputs, train_targets, eval_targets, train_lang_codes, eval_lang_codes = train_test_split(
    inputs, targets, lang_codes, test_size=0.1
)

# Tokenize train and eval data
train_encodings = tokenizer(train_inputs, max_length=128, truncation=True, padding="max_length")
train_labels = tokenizer(train_targets, max_length=128, truncation=True, padding="max_length")

eval_encodings = tokenizer(eval_inputs, max_length=128, truncation=True, padding="max_length")
eval_labels = tokenizer(eval_targets, max_length=128, truncation=True, padding="max_length")

# Convert to dataset format
train_dataset = Dataset.from_dict({
    "input_ids": train_encodings["input_ids"],
    "attention_mask": train_encodings["attention_mask"],
    "labels": train_labels["input_ids"]
})

eval_dataset = Dataset.from_dict({
    "input_ids": eval_encodings["input_ids"],
    "attention_mask": eval_encodings["attention_mask"],
    "labels": eval_labels["input_ids"]
})

# Define training parameters
training_args = TrainingArguments(
    output_dir="./nllb_bn",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,  # Adjust for GPU memory
    per_device_eval_batch_size=4,
    num_train_epochs=3,  # Increase for better fine-tuning
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
    logging_dir="./logs",
    logging_steps=500,
    report_to="none",
    remove_unused_columns=False  # 🚀 Prevents missing columns error
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Ensure GPU is being used if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
trainer.train()

# Evaluate model performance
trainer.evaluate()

# Save fine-tuned model
model.save_pretrained("./nllb_bn_finetuned")
tokenizer.save_pretrained("./nllb_bn_finetuned")

# Load the fine-tuned model to verify saving was successful
model_name = "./nllb_bn_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Function for translation inference
def translate_text(text, src_lang="eng_Latn", tgt_lang="ben_Beng"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    output_ids = model.generate(**inputs)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translated_text

# Test the fine-tuned model with a sample sentence
sample_text = "Hello, how are you?"
translated_text = translate_text(sample_text)
print("Translated Text:", translated_text)

# Push to Hugging Face Hub (Optional)
# from huggingface_hub import notebook_login
# notebook_login()
# model.push_to_hub("your-hf-username/nllb_bn_finetuned")
# tokenizer.push_to_hub("your-hf-username/nllb_bn_finetuned")
