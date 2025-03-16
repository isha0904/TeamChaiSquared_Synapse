## ReviewVaani - A new-age tool to make shopping more accessible for all

Our Amazon Product Review Translator is an AI-powered tool that translates customer reviews into multiple Indian languages to enhance accessibility and improve purchase decisions. 
The system leverages state-of-the-art NLP models for accurate, context-aware translations while maintaining the original sentiment and intent of the review.

We have pushed our model to HuggingFace: https://huggingface.co/447AnushkaD/nllb_bn_finetuned

TeamChaiSquared_Synapse/
│── static/                  # Static files for frontend  
│   ├── css/  
│   ├── js/  
│   ├── assets/logo.png, penguin.png
│
│── template/                # HTML templates for the UI  
│   ├── product.html  
│   
│── app.py                   # Main backend application (Flask/FastAPI)  
│── README.md                # Project documentation  

│── model/                  # Codes for building, training model and pushing to Huggingface
│   ├── model.py 
│   ├── upload_hf.py

