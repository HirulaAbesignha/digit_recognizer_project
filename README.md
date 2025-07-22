# Digit Recognizer Project

This project aims to classify handwritten digits from the popular [Kaggle Digit Recognizer dataset](https://www.kaggle.com/c/digit-recognizer), which is a subset of the MNIST dataset. The objective is to build and train a machine learning model that can accurately predict digits (0–9) from image pixel data.

## 📂 Project Structure

```
digit_recognizer_project/
│
├── digit_model.pkl             # Trained model saved for reuse
├── digit_recognizer.ipynb      # Main notebook for data processing and training
├── train.csv                   # Training dataset from Kaggle
├── test.csv                    # Test dataset (optional use)
├── predict_sample.csv          # Sample predictions output (optional)
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation
```

## 📊 Dataset

- **Source:** [Kaggle Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
- **Format:** CSV files
- **Content:**
  - Each image is 28x28 pixels, flattened into a vector of 784 features.
  - The training set includes labels (0–9) for supervised learning.

## 🔧 Features

- Data preprocessing with NumPy and Pandas
- Visualization using Matplotlib
- Model training using Logistic Regression and/or other classifiers
- Accuracy evaluation on validation data
- Model serialization using `pickle`
- Custom digit prediction pipeline

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/HirulaAbesignha/digit_recognizer_project.git
cd digit_recognizer_project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

You can open and run `digit_recognizer.ipynb` using Jupyter Notebook or any IDE that supports notebooks (e.g., VS Code, Colab).

### 4. Predict using the trained model

After training, the model is saved as `digit_model.pkl`. You can load it and use it to make predictions on new digit images.

## 📈 Model Performance

- Achieved ~**XX% accuracy** on the validation dataset  
  _(Replace this with your actual score if available.)_

## 🛠️ Tools & Libraries

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- pickle

## 📌 Future Improvements

- Implement CNN with TensorFlow/Keras for higher accuracy
- Add GUI for digit prediction from drawn input
- Deploy as a web app using Flask or Streamlit

## 🧠 Inspiration

This project is inspired by the classic MNIST digit classification problem and serves as a foundational step into computer vision and machine learning.

## 📬 Contact

**Hirula Abesignha**  
[GitHub Profile](https://github.com/HirulaAbesignha)

---

*Feel free to fork and contribute to this project!*
