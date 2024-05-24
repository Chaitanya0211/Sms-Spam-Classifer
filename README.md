# Email/SMS Spam Classifier

This is a simple web application for classifying email and SMS messages as spam or not spam using a machine learning model. The app is built with Streamlit and uses a pre-trained model to make predictions.

## Features

- Input email or SMS text to classify as spam or not spam
- Load example messages to test the classifier
- Clear input field to reset the form
- Provides feedback on the classification result

## Screenshots

![Screenshot 1](Screenshot1.png)
![Screenshot 2](Screenshot2.png)

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/spam-classifier.git
    cd spam-classifier
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure that you have NLTK's stopwords and punkt tokenizer downloaded:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

5. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Open your web browser and navigate to `http://localhost:8501`.
2. Enter the email or SMS text you want to classify in the input box.
3. Click on the `Predict` button to see if the message is classified as spam or not spam.
4. Use the `Clear` button to reset the input field.
5. You can also load example messages from the sidebar to see how they are classified.

## Project Structure

- `app.py`: The main application script.
- `vectorizer.pkl`: The saved TF-IDF vectorizer.
- `model.pkl`: The pre-trained machine learning model.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## Dependencies

- streamlit
- scikit-learn
- nltk
- pickle

You can install the dependencies using:
```bash
pip install -r requirements.txt
