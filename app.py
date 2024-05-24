import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time

nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Email/SMS Spam Classifier", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://www.transparenttextures.com/patterns/clean-gray-paper.png');
        background-size: cover;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
        transition-duration: 0.4s;
    }
    .stButton>button:hover {
        background-color: white; 
        color: black; 
        border: 2px solid #4CAF50;
    }
    .stTextArea textarea {
        font-size: 18px;
        height: 200px;
    }
    .stSidebar .stRadio div {
        padding: 10px;
    }
    .stHeader h1 {
        color: #4CAF50;
        font-family: 'Courier New', Courier, monospace;
    }
    .stHeader h2 {
        color: #4CAF50;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for input_sms
if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

# Sidebar
st.sidebar.title("Email/SMS Spam Classifier")
st.sidebar.markdown(
    """
    **Instructions:**
    1. Enter the email or SMS text in the input box.
    2. Click on the 'Predict' button to classify the text as Spam or Not Spam.
    3. Click on the 'Clear' button to reset the input field.
"""
)

# Main panel
st.title("Email/SMS Spam Classifier")

# Columns for layout
col1, col2 = st.columns(2)

with col1:
    st.header("Enter the message")
    input_sms = st.text_area("", value=st.session_state.input_sms)

    if st.button("Clear"):
        st.session_state.input_sms = ""
        st.experimental_rerun()

# Example messages section
st.sidebar.header("Example Messages")
example_msgs = [
    "Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 87121 to receive entry question(std txt rate)",
    "Nah I don't think he goes to usf, he lives around here though",
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
]

if st.sidebar.button("Load Example 1"):
    st.session_state.input_sms = example_msgs[0]
    st.experimental_rerun()
if st.sidebar.button("Load Example 2"):
    st.session_state.input_sms = example_msgs[1]
    st.experimental_rerun()
if st.sidebar.button("Load Example 3"):
    st.session_state.input_sms = example_msgs[2]
    st.experimental_rerun()

if st.button("Predict"):
    with col2:
        if input_sms.strip() != "":
            # Show a loader animation
            with st.spinner("Predicting..."):
                time.sleep(2)  # Simulate a delay

            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict
            result = model.predict(vector_input)[0]
            # 4. Display
            if result == 1:
                st.header("Spam")
                st.write("The entered text is classified as **Spam**.")
            else:
                st.header("Not Spam")
                st.write("The entered text is classified as **Not Spam**.")
            st.success("Prediction complete!")
        else:
            st.warning("Please enter a message to classify.")

# Feedback Section
st.sidebar.title("Feedback")
st.sidebar.write("Was the classification accurate?")

feedback = st.sidebar.radio("Please select:", ("Yes", "No"))

if feedback == "Yes":
    st.sidebar.success("Thank you for your feedback!")
else:
    st.sidebar.warning(
        "Thank you for your feedback! We will work on improving the model."
    )
