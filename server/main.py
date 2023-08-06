from transformers import pipeline
from flask import Flask
app = Flask(__name__)


if __name__ == '__main__':
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")
    sequence_to_classify = "one day I will see the world"
    candidate_labels = ['travel', 'cooking', 'dancing']
    print(classifier(sequence_to_classify, candidate_labels))
    app.run(host='0.0.0.0', port=5000, debug=True)
