from flask import Flask, request, jsonify
from transformers import NllbTokenizerFast, AutoModel
import torch

app = Flask(__name__)

# Model Name
MODEL_NAME = "447AnushkaD/nllb_bn_finetuned"

# Explicitly use NllbTokenizerFast instead of AutoTokenizer
tokenizer = NllbTokenizerFast.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    return jsonify({"output": outputs.last_hidden_state.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
