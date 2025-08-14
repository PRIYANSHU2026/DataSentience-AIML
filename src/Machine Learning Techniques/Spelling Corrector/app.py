"""
Spelling Corrector - Streamlit Application
"""
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import sys
import re
import unidecode
from pathlib import Path

# Add parent directory to path to import model utilities
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set page config
st.set_page_config(
    page_title="Spelling Corrector",
    page_icon="✏️",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #1e88e5;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .input-section {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .word-box {
        display: inline-block;
        margin: 0.2rem;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        background-color: #bbdefb;
        font-weight: 500;
    }
    .correction {
        color: #2e7d32;
        font-weight: 600;
    }
    .original {
        color: #c62828;
        text-decoration: line-through;
        margin-right: 0.5rem;
    }
    .model-info {
        background-color: #f1f8e9;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
SOS = '\t'  # start of sequence
EOS = '*'    # end of sequence
CHARS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ')
REMOVE_CHARS = r'[#$%"\+@<=>!&,-.?:;()*\[\]^_`{|}~/\d\t\n\r\x0b\x0c]'

class CharacterTable:
    """Character-level one-hot encoding/decoding."""
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char2index = {c: i for i, c in enumerate(self.chars)}
        self.index2char = {i: c for i, c in enumerate(self.chars)}
        self.size = len(self.chars)
    
    def encode(self, C, nb_rows):
        """One-hot encode given string C."""
        x = np.zeros((nb_rows, len(self.chars)), dtype=np.float32)
        for i, c in enumerate(C):
            x[i, self.char2index[c]] = 1.0
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output."""
        if calc_argmax:
            indices = x.argmax(axis=-1)
        else:
            indices = x
        chars = ''.join(self.index2char[ind] for ind in indices)
        return indices, chars

def tokenize(text):
    """Split text into tokens."""
    tokens = [re.sub(REMOVE_CHARS, '', token)
              for token in re.split(r"[-\n ]", text)]
    return [t for t in tokens if t]

def load_models():
    """Load the pre-trained models."""
    try:
        # Load the model architecture and weights
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'spelling_corrector.h5')
        model = load_model(model_path, custom_objects={
            'truncated_acc': lambda y_true, y_pred: y_true,
            'truncated_loss': lambda y_true, y_pred: y_pred
        })
        
        # Create character tables
        input_chars = set(' '.join(CHARS))
        target_chars = set(SOS + EOS + ' ' + ''.join(CHARS))
        input_ctable = CharacterTable(input_chars)
        target_ctable = CharacterTable(target_chars)
        
        return model, input_ctable, target_ctable
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def preprocess_text(text, input_ctable, maxlen=46):
    """Preprocess input text for the model."""
    # Tokenize and clean text
    tokens = tokenize(text.lower())
    
    # Add EOS padding
    processed_tokens = [token + EOS * (maxlen - len(token)) for token in tokens]
    
    # Encode tokens
    encoded_tokens = np.zeros((len(processed_tokens), maxlen, input_ctable.size))
    for i, token in enumerate(processed_tokens):
        encoded_tokens[i] = input_ctable.encode(token, maxlen)
    
    return tokens, encoded_tokens

def correct_spelling(model, input_ctable, target_ctable, text, maxlen=46):
    """Correct spelling in the given text."""
    # Preprocess input
    original_tokens, encoded_tokens = preprocess_text(text, input_ctable, maxlen)
    
    # Prepare decoder input (start with SOS)
    decoder_input = np.zeros((len(original_tokens), 1, target_ctable.size))
    decoder_input[:, 0, target_ctable.char2index[SOS]] = 1.0
    
    # Initialize states
    states_value = model.predict([encoded_tokens, decoder_input])
    
    # Generate predictions
    stop_condition = False
    decoded_tokens = [''] * len(original_tokens)
    
    for _ in range(maxlen):
        output_tokens, h, c = model.predict([encoded_tokens, decoder_input] + states_value)
        
        # Get the most likely next character
        next_indices = output_tokens.argmax(axis=-1)
        
        # Update decoded tokens
        for i, idx in enumerate(next_indices):
            next_char = target_ctable.index2char.get(idx[0], '')
            if next_char != EOS:
                decoded_tokens[i] += next_char
        
        # Check if all sequences have ended
        if all(idx[0] == target_ctable.char2index[EOS] for idx in next_indices):
            break
        
        # Update decoder input for next step
        decoder_input = np.zeros((len(original_tokens), 1, target_ctable.size))
        for i, idx in enumerate(next_indices):
            if idx[0] < target_ctable.size:  # Ensure index is valid
                decoder_input[i, 0, idx[0]] = 1.0
        
        # Update states
        states_value = [h, c]
    
    return original_tokens, decoded_tokens

def main():
    """Main function to run the Streamlit app."""
    st.markdown('<h1 class="main-header">✏️ Spelling Corrector</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #424242;'>
            Enter text to correct spelling mistakes using a deep learning model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models (with caching)
    @st.cache_resource
    def load_models_cached():
        return load_models()
    
    model, input_ctable, target_ctable = load_models_cached()
    
    if model is None or input_ctable is None or target_ctable is None:
        st.error("Failed to load the spelling correction model. Please check if the model files exist.")
        return
    
    # Input section
    with st.container():
        st.markdown("### 📝 Enter Text to Correct")
        user_input = st.text_area(
            "Type or paste your text here:",
            height=150,
            placeholder="Enter text with potential spelling mistakes...",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            max_length = st.slider(
                "Maximum word length to process",
                min_value=10,
                max_value=50,
                value=30,
                help="Longer words will be truncated"
            )
        with col2:
            show_original = st.checkbox(
                "Show original words",
                value=True,
                help="Display original words alongside corrections"
            )
        
        process_button = st.button("🔍 Correct Spelling", use_container_width=True)
    
    # Process and display results
    if process_button and user_input.strip():
        with st.spinner("Correcting spelling..."):
            try:
                original_tokens, corrected_tokens = correct_spelling(
                    model, input_ctable, target_ctable, user_input, max_length)
                
                if not corrected_tokens or all(not token for token in corrected_tokens):
                    st.warning("No corrections were made. The text may already be correct or contain no valid words.")
                else:
                    # Display results
                    st.markdown("### ✅ Corrected Text")
                    corrected_text = []
                    
                    for orig, corr in zip(original_tokens, corrected_tokens):
                        if orig.lower() != corr.lower() and corr.strip():
                            if show_original:
                                corrected_text.append(f"<span class='original'>{orig}</span><span class='correction'>{corr}</span>")
                            else:
                                corrected_text.append(f"<span class='correction'>{corr}</span>")
                        else:
                            corrected_text.append(orig)
                    
                    st.markdown(
                        "<div style='line-height: 2.5;'>" + 
                        " ".join(corrected_text) + 
                        "</div>", 
                        unsafe_allow_html=True
                    )
                    
                    # Show detailed corrections
                    st.markdown("### 📊 Correction Details")
                    corrections = [(o, c) for o, c in zip(original_tokens, corrected_tokens) 
                                 if o.lower() != c.lower() and c.strip()]
                    
                    if corrections:
                        cols = st.columns(3)
                        for i, (orig, corr) in enumerate(corrections):
                            with cols[i % 3]:
                                st.markdown(
                                    f"<div class='word-box'>{orig} → <strong>{corr}</strong></div>",
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("No spelling corrections were needed for this text.")
                
            except Exception as e:
                st.error(f"An error occurred during spelling correction: {str(e)}")
    
    # Add information about the model
    with st.expander("ℹ️ About This Model", expanded=False):
        st.markdown("""
        This spelling corrector uses a sequence-to-sequence deep learning model 
        trained on a large corpus of text with common spelling mistakes.
        
        **How it works:**
        1. The input text is split into individual words
        2. Each word is processed by the neural network
        3. The model predicts the most likely correct spelling
        4. The corrected words are combined back into the text
        
        **Tips for best results:**
        - Keep sentences under 100 words for optimal performance
        - The model works best with common words and phrases
        - Proper nouns and technical terms may not be corrected accurately
        """)
    
    # Add a footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem; margin-top: 2rem;'>
        <p>This spelling corrector is powered by a deep learning model.</p>
        <p>For best results, use complete words and proper punctuation.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
import tensorflow as tf
import joblib

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('./Model/seq2seq_epoch_100.h5')
  return model
def load_transformer():
  transformer = joblib.load("./Model/data_transformer.joblib")
  return transformer
with st.spinner('Model is being loaded..'):
  model=load_model()
  transformer=load_transformer()

st.title('Spell Checker Using Sequence to Sequence Model')

text = st.text_area("Enter Text:", value='', height=None, max_chars=None, key=None)

if st.button('Correct Spelling'):
    if text == '':
        st.write('Please enter text for checking') 
    else: 
        prediction = model.predict(transformer.transform(text))
        corrected_spell=prediction[0]
        st.write('Corrected Word - ' + str(corrected_spell))
else: pass
