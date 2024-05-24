import pickle
import string
from flask import Flask, request, render_template, redirect, url_for
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import time

nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__, template_folder="template")
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


# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

example_msgs = [
    "Free entry in 2 a weekly competition to win FA Cup final tickets. Text FA to 87121 to receive entry question(std txt rate)",
    "Nah I don't think he goes to usf, he lives around here though",
    "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
]


@app.route("/", methods=["GET", "POST"])
def home():
    input_sms = ""
    classification = None

    if request.method == "POST":
        input_sms = request.form.get("input_sms", "")

        if input_sms.strip() != "":
            # Simulate a delay
            time.sleep(2)

            # Preprocess
            transformed_sms = transform_text(input_sms)
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            # Predict
            result = model.predict(vector_input)[0]

            if result == 1:
                classification = "Spam"
            else:
                classification = "Not Spam"

    return render_template(
        "index.html", input_sms=input_sms, classification=classification
    )


@app.route("/clear", methods=["POST"])
def clear():
    return redirect(url_for("home"))


@app.route("/load_example/<int:example_id>", methods=["POST"])
def load_example(example_id):
    if 1 <= example_id <= len(example_msgs):
        input_sms = example_msgs[example_id - 1]
        return render_template("index.html", input_sms=input_sms, classification=None)
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
