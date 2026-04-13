import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- PAGE CONFIG --------------------
st.set_page_config(layout="wide")

# -------------------- UI DESIGN --------------------
st.markdown("""
<style>
.header-container {
    text-align: center;
    padding-top: 20px;
    padding-bottom: 10px;
}

.gradient-text {
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    font-size: 18px;
    font-style: italic;
    color: white;
    margin-top: 10px;
}

div.stButton > button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: black;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}
</style>

<div class="header-container">
    <div class="gradient-text">🌐 AI Multilingual Translator</div>
    <div class="subtitle">Real-time English ↔ Hindi Translation</div>
</div>
""", unsafe_allow_html=True)

# -------------------- HISTORY --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    en_hi_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
    en_hi_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

    hi_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
    hi_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-hi-en")

    return en_hi_tokenizer, en_hi_model, hi_en_tokenizer, hi_en_model

en_hi_tokenizer, en_hi_model, hi_en_tokenizer, hi_en_model = load_models()

# -------------------- INPUT --------------------
st.markdown("### 📝 Enter Text")
text = st.text_area("Enter Text", label_visibility="collapsed")

st.markdown("### 🌐 Select Translation")
direction = st.selectbox(
    "Select Language",
    ["English → Hindi", "Hindi → English"],
    label_visibility="collapsed"
)

# -------------------- TRANSLATE FUNCTION --------------------
def translate(text, direction):
    if direction == "English → Hindi":
        inputs = en_hi_tokenizer(text, return_tensors="pt", padding=True)
        outputs = en_hi_model.generate(**inputs)
        return en_hi_tokenizer.decode(outputs[0], skip_special_tokens=True)

    elif direction == "Hindi → English":
        inputs = hi_en_tokenizer(text, return_tensors="pt", padding=True)
        outputs = hi_en_model.generate(**inputs)
        return hi_en_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------- BUTTON --------------------
if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        result = translate(text, direction)

        st.session_state.history.append({
            "input": text,
            "direction": direction,
            "output": result
        })

        st.success(result)

# -------------------- HISTORY DISPLAY --------------------
st.sidebar.title("🕘 History")

if st.sidebar.button("Clear History"):
    st.session_state.history = []

for item in reversed(st.session_state.history):
    st.sidebar.write(f"**{item['direction']}**")
    st.sidebar.write(item["input"])
    st.sidebar.write(f"➡ {item['output']}")
    st.sidebar.write("---")