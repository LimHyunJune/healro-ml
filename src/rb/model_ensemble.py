import xmlrpc.client

class ModelEnsemble:
    def __init__(self):
        self.proxy = xmlrpc.client.ServerProxy("http://localhost:8000")

    def predict(self, sentence):
        output_1 = self.proxy.predict_bart_large_mnli(sentence)
        output_2 = self.proxy.predict_roberta_large_mnli(sentence)
        output_3 = self.proxy.predict_deberta_large_mnli(sentence)
        print(output_1)
        print(output_2)
        print(output_3)

if __name__ == "__main__":
    me = ModelEnsemble()
    me.predict("I hate server proxy.")
