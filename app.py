import streamlit as st
import numpy as np 
import pickle 
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Next Word Predictor", layout='centered')
st.markdown("""
<style>
    .stTextInput input {
        font-size: 18px;
    }
    .suggestion-text {
        font-size: 18px;
        color: #1E88E5;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

#1. Load Resources 
@st.cache_resource
def load_resources():
    try : 
        model = load_model("models/model_20_25.keras")

        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        return model, tokenizer
    except Exception as e : 
        return None, None

model, tokenizer = load_resources()


#2. Predition Function

def predict_next_words(text):
    if not text or text.strip() == '':
        return None

    #Preparing input
    max_sequence_len = model.input_shape[1] + 1
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    #Predict
    probs = model.predict(token_list, verbose=0)[0]

    #Smart Choice : Pick randomly from the top 3 candidates to avoid loops 
    top_indices = probs.argsort()[-3:][::-1]

    suggestions = []
    for index in top_indices:
        word = tokenizer.index_word.get(index, '')
        if word and word != '<OOV>': 
            suggestions.append(word)
    return suggestions

#3. CallBack Function 
def append_word(word_to_append): 
    current_text = st.session_state.input_box
    
    if current_text.endswith(' '): 
        st.session_state.input_box = current_text + word_to_append
    else : 
        st.session_state.input_box = current_text + ' ' + word_to_append


#4. UI
st.title("Next Word Predictor")
st.subheader("AI Autocomplete")
st.write("Type a sentence, and I'll guess what comes next.")

if 'input_box' not in st.session_state:
    st.session_state.input_box = ''

user_text = st.text_input('Start typing....', key='input_box', placeholder='e.g. I am skilled....')


if user_text:
    suggestions = predict_next_words(user_text)
    if suggestions:
        st.write('### Suggestions : ')
        cols = st.columns(3)
        for i, word in enumerate(suggestions):
            with cols[i]:
                st.button(
                    word, 
                    on_click=append_word, 
                    args=(word,), 
                    key=f'btn_{i}', 
                    use_container_width=True
                    )

st.markdown("---")
with st.expander("How it works"):
    st.write("This model uses a **Stacked LSTM** trained on 160k+ conversations.")
    st.write("It uses **Temperature Sampling** to avoid repetitive loops.")    