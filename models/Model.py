
class Model:
    model = None

    en_preprocess = None
    fr_preprocess = None

    en_data = None
    fr_data = None

    def __init__(self, en_preprocess, fr_preprocess):
        self.en_preprocess = en_preprocess
        self.fr_preprocess = fr_preprocess

    def build_model(self):
        pass

    def fit_model(self):
        pass

    def predict(self, sentences):
        return self.model.predict(sentences, len(sentences))

    def summary(self):
        self.model.summary()

    def save_model_weights(self, path):
        self.model.save_weights(path)
        print("model weights saved under", path)

    def load_model_weights(self, path):
        self.model.load_weights(path)
        print("model weights loaded from", path)