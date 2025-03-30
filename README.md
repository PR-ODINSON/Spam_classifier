# Spam Classifier using LSTM

## Overview
This project is a spam classifier built using a Bidirectional LSTM model. It processes text messages and classifies them as spam or ham (not spam). The dataset is preprocessed, tokenized, padded, and trained using a deep learning model. 

## Features
- Text preprocessing (removal of special characters, stopwords, and lowercasing)
- Tokenization and padding
- Handling class imbalance using SMOTE
- Deep learning model using Bidirectional LSTMs
- Early stopping to prevent overfitting
- Model evaluation and prediction on test data

## Dependencies
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy tensorflow scikit-learn nltk imbalanced-learn
```

## Dataset
The model is trained on a CSV file named `combined_data.csv` containing text messages labeled as spam (1) or ham (0). Make sure to place the dataset in the same directory as the script.

## Usage
### 1. Clone the repository
```bash
git clone https://github.com/yourusername/spam-classifier-lstm.git
cd spam-classifier-lstm
```

### 2. Train the model
Run the script to train and evaluate the model:
```bash
python spam_classifier.py
```

### 3. Test with a sample email
Modify `sample_email` in the script and run:
```python
sample_email = "Urgent: Your account has been compromised. Click here to secure it!"
```
The model will predict whether the email is spam or not.

### 4. Save and Load the Model
After training, the model is saved as `spam_classifier_lstm_tuned.h5` and the tokenizer as `tokenizer_tuned.pkl`. You can reuse them without retraining.

## Results
- The model achieves high accuracy in spam detection.
- The use of SMOTE balances class distribution, improving performance on imbalanced datasets.

## Author
Developed by **Prithviraj Verma** | GitHub: [PR-ODINSON](https://github.com/PR-ODINSON)

## License
This project is open-source and available under the MIT License.
