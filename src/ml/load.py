from transformers import pipeline

class Model:
    def __init__(self):
        self.bart_large_mnli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.label = ["port", "server", "web"]

    def predict_bart_large_mnli(self, sentence):
        return self.bart_large_mnli(sentence, self.label, multi_label=True)


