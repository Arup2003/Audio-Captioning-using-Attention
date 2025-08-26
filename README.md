# Attention-Based Audio Captioning
This project is an attention-based audio captioning model built from scratch, inspired by Ikawa & Kashino (DCASE 2019).
- Instead of video → captions, we do audio → captions, which also preserves the temporal dependecies.
- Audio embeddings come from pretrained YAMNet.
- Word embeddings are initialized with GloVe.
- The model is a BiLSTM encoder + LSTM decoder with additive attention that learns to “listen” and “speak.”

The goal? See whether pretrained representations can help a small dataset model “hear” sounds like doors closing, birds chirping, or music playing and describe them in natural language.

# ⚡ Quickstart
1. Clone repo & install deps
```
git clone https://github.com/arup2003/Audio-Captioning-using-Attention.git
cd Audio-Captioning-using-Attention
pip install -r requirements.txt
```
2. To quickly use and play with the model:
```
streamlit run web_app.py
```
4. Now, for audio-caption.ipynb:  
Prepare data by placing your clotho audio clips in
```
data/audio/
```
4. Download Clotho dataset captions (already expected in `data/`)

5. Generate embeddings
To save processing time, embeddings are precomputed:
```
#inside notebook
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
#Run embedding extraction block in the notebook
```


# 🧠 Model Architecture
Our audio captioning model follows an encoder-decoder with attention paradigm, inspired by Ikawa & Kashino (DCASE 2019). The architecture bridges audio understanding and natural language generation through pretrained embeddings and attention mechanisms.

<p align="center">
  <img src="https://github.com/user-attachments/assets/866e68f3-dcc3-4812-adcd-7c27c90a3e91" alt="Architecture diagram from Ikawa & Kashino (DCASE 2019)" width="70%">
  <br>
  <em>
    Architecture diagram reproduced from Ikawa & Kashino, "Attention-based audio captioning and sound event detection with cross-task learning" (DCASE 2019).
    <a href="https://arxiv.org/abs/1907.10043">arXiv:1907.10043</a>
  </em>
</p>



- Bidirectional LSTM encoder (512 units per direction) processing 1024‑dimensional audio embeddings from YAMNet.
- LSTM decoder (512 units) generating captions word-by-word.
- Additive (Bahdanu) attention aligns decoder steps with relevant audio features.
- Pretrained GloVe vectors (300 dimensions) provide semantic context.
- Time-distributed dense output layer predicts tokens.
- Dropout applied throughout encoder, decoder, and attention layers to reduce overfitting.

#

Thanks for checking it out!


