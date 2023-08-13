from flask import Flask, render_template
from transformers import pipeline
from rb.model_ensemble import ModelEnsemble
from ml.load import Model
import requests

app = Flask(__name__)
me = ModelEnsemble()


@app.route("/")
def predict():
    print(model.predict_deberta_large_mnli("I hate proxy server."))
    return "success"


if __name__ == "__main__":
    model = Model()
    app.run()