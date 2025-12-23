# Context-Aware Next Word Predictor

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

## Project Overview
This project implements a generative text completion model using Deep Learning (Stacked LSTMs) rather than traditional N-gram frequency tables. The model is designed to predict the next word in a sequence by analyzing long-term dependencies in multi-turn conversational data.

## Architecture
- **Model:** Stacked LSTM (Long Short-Term Memory) network with Embedding layers.
- **Training Data:** DailyDialog dataset (160,000+ multi-turn conversations).
- **Tokenizer:** Custom Keras tokenizer with OOV (Out-Of-Vocabulary) token handling.
- **Inference Strategy:** Implements "Smart Sampling" (Temperature Scaling) to ensure output diversity and prevent deterministic repetition loops.

## Project Structure
NextWord-Predictor/
├── Data/                   # Raw and processed datasets
├── models/                 # Serialized model weights (.h5) and tokenizer (.pickle)
├── Model_Training.ipynb    # Jupyter notebook for data preprocessing and model training
├── app.py                  # Streamlit interface for real-time inference
└── requirements.txt        # Python dependencies

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/patlegar-manjunatha/NextWord-Predictor.git](https://github.com/patlegar-manjunatha/NextWord-Predictor.git)
   ```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Model Performance Details

The model was trained using Categorical Crossentropy loss. The preprocessing pipeline includes:

* Regex-based text sanitization.
* Dynamic sequence padding for variable-length inputs.
* Vocabulary optimization to balance model size and coverage.

---

**Author:** [Manjunatha Patlegar](https://www.linkedin.com/in/patlegar-manjunatha/)

```

```
