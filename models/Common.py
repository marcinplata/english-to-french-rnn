from models.Model import Model as RNNModel
from keras.losses import sparse_categorical_crossentropy
from keras.layers import Embedding, GRU, Dense, Bidirectional, RepeatVector, Dropout, GRUCell, RNN
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Sequential


class Common(RNNModel):

    compile_args = {
        "loss": sparse_categorical_crossentropy,
        "optimizer": Adam(0.007),
        "metrics": ['accuracy']
    }

    fit_args = {
        "batch_size": 1024,
        "epochs": 25,
        "validation_split": 0.1
    }

    input_shape = None
    output_sequence_length = None
    english_vocab_size = None
    french_vocab_size = None

    def __init__(self, en_preprocess, fr_preprocess):
        super().__init__(en_preprocess, fr_preprocess)

        # self.en_data = self.en_data.reshape((*self.en_data.shape, 1))
        self.fr_data = self.fr_data.reshape((*self.fr_data.shape, 1))

        self.input_shape = self.en_data.shape[1]
        self.output_sequence_length = self.fr_data.shape[1]
        self.english_vocab_size = len(self.en_preprocess.tokenizer.word_index) + 1
        self.french_vocab_size = len(self.fr_preprocess.tokenizer.word_index) + 1

    def build_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.english_vocab_size, 60, input_length=self.input_shape))

        self.model.add(GRU(60))

        self.model.add(RepeatVector(self.output_sequence_length))

        self.model.add(Bidirectional(GRU(70, return_sequences=True)))

        self.model.add(Dense(self.french_vocab_size))
        self.model.add(Activation("softmax"))

        self.model.compile(**self.compile_args)

    def fit_model(self):
        self.model.fit(self.en_data, self.fr_data, **self.fit_args)

