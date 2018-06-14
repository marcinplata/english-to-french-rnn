from models.Model import Model as RNNModel
from keras.losses import sparse_categorical_crossentropy
from keras.layers import Bidirectional, GRU, Dense, TimeDistributed
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Sequential


class Bidirect(RNNModel):

    compile_args = {
        "loss": sparse_categorical_crossentropy,
        "optimizer": Adam(0.05),
        "metrics": ['accuracy']
    }

    fit_args = {
        "batch_size": 1024,
        "epochs": 1,
        "validation_split": 0.2
    }

    input_shape = None
    french_vocab_size = None

    def __init__(self, en_preprocess, fr_preprocess):
        en_preprocess.pad(fr_preprocess.get_preprocessed_sequences().shape[1])

        super().__init__(en_preprocess, fr_preprocess)

        self.en_data = self.en_data.reshape((*self.en_data.shape, 1))
        self.fr_data = self.fr_data.reshape((*self.fr_data.shape, 1))

        self.input_shape = self.en_data.shape[1:]
        self.french_vocab_size = len(self.fr_preprocess.tokenizer.word_index) + 1

    def build_model(self):
        gru = GRU(100, return_sequences=True)
        self.model = Sequential()
        self.model.add(Bidirectional(gru, input_shape=self.input_shape))
        self.model.add(TimeDistributed(Dense(self.french_vocab_size)))
        self.model.add(Activation("softmax"))

        self.model.compile(**self.compile_args)

    def fit_model(self):
        self.model.fit(self.en_data, self.fr_data, **self.fit_args)

    def predict(self, sentences):
        sentences = sentences.reshape((*sentences.shape, 1))
        return self.model.predict(sentences, len(sentences))