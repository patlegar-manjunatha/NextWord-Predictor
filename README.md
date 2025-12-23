# Context-Aware Next Word Predictor

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## Project Overview
This project implements a generative text completion model using Deep Learning (Stacked LSTMs) rather than traditional N-gram frequency tables. The model is designed to predict the next word in a sequence by analyzing long-term dependencies in multi-turn conversational data.

The model was trained on the DailyDialog dataset, processing over 168,000 conversation lines to capture casual, human-like speech patterns.

## Technical Architecture
The core is a sequential neural network built with TensorFlow/Keras:

- **Embedding Layer:** Converts the 19,391-word vocabulary into dense 100-dimensional vectors.
- **LSTM Layer 1:** 150 units with return_sequences=True to capture lower-level sequence patterns.
- **Dropout:** Rate of 0.2 implemented for regularization.
- **LSTM Layer 2:** 100 units to capture higher-level semantic context.
- **Dense Output:** A softmax layer predicting the probability of the next word across the entire vocabulary.

**Training Strategy:**
The model uses Sparse Categorical Crossentropy loss. A custom training loop was implemented to save model states every 5 epochs (Total 25 epochs), allowing for granular performance monitoring and fault tolerance.

## Data Engineering Pipeline
The raw data required significant preprocessing (handled in file.ipynb):

1. **AST Parsing:** Implemented robust parsing to convert stringified lists from the raw CSV into usable text.
2. **Regex Sanitization:**
   - Fixed "fused" sentences (e.g., separating "time?I" into "time? I").
   - Normalized detached contractions (e.g., joining "It ' s" into "It's").
   - Removed non-English artifacts and noise.
3. **Dynamic Padding:** Sequences were padded to a length of 68 tokens based on the dataset distribution.

## Inference Engine (Smart Sampling)
To prevent deterministic repetition loops common in LSTM models, the inference engine in app.py implements a Top-K Sampling Strategy.

Instead of selecting the single highest probability word, the system selects from the Top 3 most probable words and samples randomly among them. This results in more diverse and natural sentence completions.

## Repository Structure
NextWord-Predictor/
├── Data/
│   ├── final_training_data_refined.txt  # Cleaned dataset (168k lines)
│   └── (Raw CSVs)                       # Original DailyDialog files
├── models/
│   ├── model_20_25.keras                # Final trained model (Epoch 25)
│   └── tokenizer.pickle                 # Serialized tokenizer (Vocab: 19,391)
├── Model_Training.ipynb                 # Architecture definition and training loop
├── file.ipynb                           # Data cleaning and preprocessing pipeline
├── app.py                               # Streamlit Web App (Inference Interface)
└── requirements.txt                     # Dependencies

## Installation and Usage

1. Clone the repository:
   git clone https://github.com/patlegar-manjunatha/NextWord-Predictor.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Interface:
   streamlit run app.py

---
**Author:** [Manjunatha Patlegar](https://www.linkedin.com/in/patlegar-manjunatha/)
