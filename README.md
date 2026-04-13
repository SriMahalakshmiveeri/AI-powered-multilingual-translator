AI-powered multilingual translator that translates text between English, Telugu, and Hindi using pretrained transformer models.
For the frontend, I used Streamlit, which allows me to build a simple and interactive web interface. I also used HTML and CSS to improve the UI design. In the interface, the user can enter text and select the translation direction.
For the backend, I used Python, where I implemented the main logic of the application. When the user clicks the translate button, the input text is sent to the backend function.

For the machine learning part, I used pretrained transformer models from Hugging Face. Specifically:
    I used Helsinki-NLP/opus-mt-en-hi for English to Hindi
    I used Helsinki-NLP/opus-mt-hi-en for Hindi to English
    I used facebook/nllb-200-distilled-600M for English and Telugu translation
These models are based on Seq2Seq (Sequence-to-Sequence) architecture, which converts input text into tokens, processes it, and generates translated output.
I used PyTorch as the backend framework because these models run on PyTorch. It helps in handling model loading, inference, and tensor operations.

Technically, the flow works like this:

    User enters text in Streamlit UI
    Streamlit sends the input to a Python function
    The function selects the correct model based on language
    The text is tokenized using AutoTokenizer
    The model generates output using the .generate() method
    The output tokens are decoded into readable text
    The result is displayed back in the UI
    
I also implemented history management using Streamlit session state, where users can view, rename, or delete previous translations.
For development and testing, I used Google Colab to load and test the models, and then integrated them into my local project using VS Code.
The system provides real-time translation with good accuracy, and since I used pretrained models, I didn’t need to train from scratch, which saved time and improved performance.
