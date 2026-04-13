import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

st.markdown("""
<style>
/* Center container */
.header-container {
    text-align: center;
    padding-top: 20px;
    padding-bottom: 10px;
}

/* Gradient Title */
.gradient-text {
    font-size: 48px;
    font-weight: bold;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Subtitle */
.subtitle {
    font-size: 18px;
    font-style: italic;
    color: white;
    margin-top: 10px;
}

/* Logo style */
.logo {
    font-size: 60px;
    margin-bottom: 10px;
}

                /* -------- TEXT AREA LABEL (Enter Text) -------- */
label[data-testid="stWidgetLabel"] {
    font-size: 20px !important;
    font-weight: 600;
}

/* -------- SELECT BOX LABEL -------- */
div[data-testid="stSelectbox"] label {
    font-size: 20px !important;
    font-weight: 600;
}

                /* ---------- ALL BUTTONS GRADIENT ---------- */

div.stButton > button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: Black;
    border: none;
    border-radius: 10px;
    height: 2em;
    font-size: 20px;
    font-weight: 800;
    padding: 0 20px; 
    transition: 0.3s ease;
}

/* Hover effect */
div.stButton > button:hover {
    background: linear-gradient(90deg, #43e97b, #38f9d7);
    color: Black;
}

/* Sidebar buttons also covered */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: Black;
}

/* Sidebar hover */
section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(90deg, #43e97b, #38f9d7);
}

</style>

<div class="header-container">
    <div class="gradient-text">🌐AI-Powered Multilingual Translator</div>
    <div class="subtitle">Breaking language barriers with intelligent translation</div>
</div>
""", unsafe_allow_html=True)


st.set_page_config(layout="wide")



# -------------------- HISTORY INIT --------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "selected_history" not in st.session_state:
    st.session_state.selected_history = None

# -------------------- SIDEBAR --------------------
st.sidebar.title("🕘 History")

if st.sidebar.button("Clear History"):
    st.session_state.history = []
    st.session_state.selected_history = None

# Show history in sidebar
for i, item in enumerate(reversed(st.session_state.history)):
    index = len(st.session_state.history) - 1 - i  # actual index

    col1, col2 = st.sidebar.columns([4, 1])

    # Use custom title if exists
    title = item.get("title", item["input"][:20] + "...")

    # Click history
    if col1.button(title, key=f"hist_{i}"):
        st.session_state.selected_history = item

    # Three dots menu
    if col2.button("⋮", key=f"menu_{i}"):
        st.session_state[f"show_menu_{i}"] = not st.session_state.get(f"show_menu_{i}", False)

    # MENU OPTIONS
    if st.session_state.get(f"show_menu_{i}", False):
        # -------- RENAME --------
        new_name = st.sidebar.text_input(
            "Rename",
            value=item.get("title", ""),
            key=f"rename_input_{i}"
        )

        if st.sidebar.button("Save", key=f"save_{i}"):
            st.session_state.history[index]["title"] = new_name
            st.session_state[f"show_menu_{i}"] = True   # keep menu open
            st.rerun()

        # -------- DELETE --------
        if st.sidebar.button("Delete", key=f"delete_{i}"):
            st.session_state.history.pop(index)
            st.session_state.selected_history = None
            st.session_state[f"show_menu_{i}"] = False  # close menu
            st.rerun()

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    multi_path = os.path.join(BASE_DIR, "translator_models", "models", "multi")
    en_hi_path = os.path.join(BASE_DIR, "translator_models", "models", "en_hi")
    hi_en_path = os.path.join(BASE_DIR, "translator_models", "models", "hi_en")


    multi_tokenizer = AutoTokenizer.from_pretrained(multi_path)
    multi_model = AutoModelForSeq2SeqLM.from_pretrained(multi_path)
    
    en_hi_tokenizer = AutoTokenizer.from_pretrained(en_hi_path)
    en_hi_model = AutoModelForSeq2SeqLM.from_pretrained(en_hi_path)

    hi_en_tokenizer = AutoTokenizer.from_pretrained(hi_en_path)
    hi_en_model = AutoModelForSeq2SeqLM.from_pretrained(hi_en_path)



    return multi_tokenizer, multi_model,en_hi_tokenizer, en_hi_model, hi_en_tokenizer, hi_en_model


multi_tokenizer, multi_model, en_hi_tokenizer, en_hi_model, hi_en_tokenizer, hi_en_model = load_models()

# -------------------- INPUT --------------------
# -------- PREMIUM LABEL: ENTER TEXT --------
st.markdown("<h4 style='margin-bottom:5px;'>📝 Enter Text</h4>", unsafe_allow_html=True)
text = st.text_area("Enter Text", label_visibility="collapsed")

# -------- PREMIUM LABEL: SELECT TRANSLATION --------
st.markdown("<h4 style='margin-bottom:5px;'>🌐 Select Translation</h4>", unsafe_allow_html=True)
direction = st.selectbox(
    "Select Language",
    ["English → Telugu", "Telugu → English", "English → Hindi", "Hindi → English"],
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

    elif direction == "English → Telugu":
        multi_tokenizer.src_lang = "eng_Latn"
        inputs = multi_tokenizer(text, return_tensors="pt")
        outputs = multi_model.generate(
            **inputs,
            forced_bos_token_id=multi_tokenizer.convert_tokens_to_ids("tel_Telu")
        )
        return multi_tokenizer.decode(outputs[0], skip_special_tokens=True)

    elif direction == "Telugu → English":
        multi_tokenizer.src_lang = "tel_Telu"
        inputs = multi_tokenizer(text, return_tensors="pt")
        outputs = multi_model.generate(
            **inputs,
            forced_bos_token_id=multi_tokenizer.convert_tokens_to_ids("eng_Latn")
        )
        return multi_tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------- TRANSLATE BUTTON --------------------
if st.button("Translate", type="primary"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        result = translate(text, direction)

        # Save to history
        st.session_state.history.append({
            "input": text,
            "direction": direction,
            "output": result
        })

        st.success(result)

# -------------------- DISPLAY SELECTED HISTORY --------------------
if st.session_state.selected_history:
    st.subheader("Selected History")
    item = st.session_state.selected_history
    st.write(f"**Direction:** {item['direction']}")
    st.write(f"**Input:** {item['input']}")
    st.write(f"**Output:** {item['output']}")