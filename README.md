## ReviewVaani - A new-age tool to make shopping more accessible for all

Our Amazon Product Review Translator is an AI-powered tool that translates customer reviews into multiple Indian languages to enhance accessibility and improve purchase decisions. 
The system leverages state-of-the-art NLP models for accurate, context-aware translations while maintaining the original sentiment and intent of the review.

We have pushed our model to HuggingFace: https://huggingface.co/447AnushkaD/nllb_bn_finetuned

[PPT Link](https://www.canva.com/design/DAGh4fpgqOc/9bsOes6roTVcWTa9KBCciA/view?utm_content=DAGh4fpgqOc&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h516fce2555)

[Video demo](https://drive.google.com/file/d/1sABelISu4KBUNOCN9DPvSfkx-7I67iO1/view?usp=drive_link)

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
