from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from transformers import pipeline


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ("/RPC2",)


class Model:
    def __init__(self):
        self.bart_large_mnli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.roberta_large_mnli = pipeline('zero-shot-classification', model='roberta-large-mnli')
        self.deberta_large_mnli = pipeline('zero-shot-classification',
                                           model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
        self.label = ["port", "server", "web"]

    def predict_bart_large_mnli(self, sentence):
        return self.bart_large_mnli(sentence, self.label, multi_label=True)

    def predict_roberta_large_mnli(self, sentence):
        return self.roberta_large_mnli(sentence, self.label, multi_label=True)

    def predict_deberta_large_mnli(self, sentence):
        return self.deberta_large_mnli(sentence, self.label, multi_label=True)


if __name__ == "__main__":
    with SimpleXMLRPCServer(('localhost', 8000), requestHandler=RequestHandler) as server:
        server.register_introspection_functions()

    server.register_instance(Model())
    print("ML load...")
    server.serve_forever()
