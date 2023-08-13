from flask import Flask, render_template
from transformers import pipeline
from ml.load import Model
import requests

app = Flask(__name__)

@app.route("/")
def predict():
    return model.predict_deberta_large_mnli("I hate proxy server.")


if __name__ == "__main__":
    model = Model()
    app.run(host='0.0.0.0')