from flask import Flask, render_template
from transformers import pipeline

app = Flask(__name__)

@app.route("/")
def index():
    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    print(classifier(sequence_to_classify, candidate_labels))
    return render_template('./index.html')