from flask import Flask, render_template
from transformers import pipeline
from rb.model_ensemble import ModelEnsemble
import requests

app = Flask(__name__)
me = ModelEnsemble()


@app.route("/predict")
def predict():
    symptom = requests.get_json()
    me.predict(symptom)
    return "success"