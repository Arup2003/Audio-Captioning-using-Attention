import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import tempfile

# Set page config
st.set_page_config(
    page_title="Audio Caption Generator",
    page_icon="🎵",
    layout="centered"
)

@st.cache_resource
def load_model_and_tokenizer():
    try:
        model = tf.keras.models.load_model('audio_captioning_model_best.keras')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            
        with open('word_mappings.pkl', 'rb') as f:
            word_mappings = pickle.load(f)
            
        return model, tokenizer, word_mappings
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def load_audio_for_yamnet(audio_file, max_duration=20):
    SAMPLE_RATE = 16000
    audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True)
    
    max_samples = SAMPLE_RATE * max_duration
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    else:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode="constant")
    
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    return tf.constant(audio, dtype=tf.float32)

@st.cache_resource
def load_yamnet():
    import tensorflow_hub as hub
    return hub.load('https://tfhub.dev/google/yamnet/1')

def extract_audio_features(audio_tensor, yamnet_model):
    _, embeddings, _ = yamnet_model(audio_tensor)
    return embeddings.numpy()

def generate_caption(audio_embedding, model, tokenizer, word_mappings, max_len=20):
    """Generate caption from audio embedding"""
    word2idx, idx2word = word_mappings
    
    sos_token = word2idx.get('<start>', word2idx.get('sos', 1))
    eos_token = word2idx.get('<end>', word2idx.get('eos', 2))
    
    caption = [sos_token]
    audio_batch = np.expand_dims(audio_embedding, 0)
    
    for _ in range(max_len):
        caption_input = np.array([caption])
        caption_padded = pad_sequences([caption], maxlen=max_len, padding="post")
        
        predictions = model.predict([audio_batch, caption_padded], verbose=0)
        next_token = np.argmax(predictions[0, len(caption)-1, :])
        
        if next_token == eos_token:
            break
            
        caption.append(next_token)
    
    words = []
    for token in caption[1:]:
        if token in idx2word:
            word = idx2word[token]
            if word not in ['<start>', '<end>', '<unk>', '']:
                words.append(word)
    
    return ' '.join(words)

st.title("🎵 Audio Caption Generator")
st.markdown("Upload an audio file and get an AI-generated caption describing what you hear!")

with st.spinner("Loading model..."):
    model, tokenizer, word_mappings = load_model_and_tokenizer()
    yamnet_model = load_yamnet()

if model is None:
    st.error("Could not load the model. Please check your file paths.")
    st.info("Make sure you have these files in your directory:")
    st.code("""
    - audio_captioning_model_best.keras
    - tokenizer.pkl
    - word_mappings.pkl
    """)

st.success("✅ Models loaded successfully!")

uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
    help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
)

# Initialize session state for caption if it doesn't exist
if 'caption' not in st.session_state:
    st.session_state.caption = None

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    _, col, _ = st.columns([1, 1, 1])
    with col:
        if st.button("🎯 Generate Caption", type="primary"):
            # Center the spinner too
            _, spinner_col, _ = st.columns([1, 1, 1])
            with spinner_col:
                with st.spinner(""):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    audio_tensor = load_audio_for_yamnet(tmp_file_path)
                    audio_embedding = extract_audio_features(audio_tensor, yamnet_model)
                    
                    # Store caption in session state
                    st.session_state.caption = generate_caption(
                        audio_embedding, model, tokenizer, word_mappings
                    )
                    
                    os.unlink(tmp_file_path)
    
    # Display caption if it exists (spans full width)
    if st.session_state.caption:
        st.markdown(
            f"""
            <div style="text-align: center;">
                <i><h1 style="font-size: 2rem; margin: 20px 0;">
                    "{st.session_state.caption}"
                </h1></i>
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown("<p style='text-align: center;'>Made by Arup :3</p>", unsafe_allow_html=True)