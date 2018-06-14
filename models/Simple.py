from models.Model import Model as NNModel
from keras.losses import sparse_categorical_crossentropy
from keras.layers import GRU, Dense, TimeDistributed
from keras.layers import Activation
from keras.optimizers import Adam
from keras.models import Sequential


class Simple(NNModel):

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
    output_sequence_length = None
    french_vocab_size = None

    def __init__(self, en_preprocess, fr_preprocess):
        super().__init__(en_preprocess, fr_preprocess)
        en_preprocess.pad(fr_preprocess.get_preprocessed_sequences().shape[1])
        en_data = en_preprocess.get_preprocessed_sequences()
        fr_data = fr_preprocess.get_preprocessed_sequences()

        en_data = en_data.reshape((*en_data.shape, 1))
        fr_data = fr_data.reshape((*fr_data.shape, 1))
        print(en_data.shape)
        print(fr_data.shape)

        self.input_shape = en_data.shape[1:]
        self.output_sequence_length = fr_data.shape[1]
        self.french_vocab_size = len(fr_preprocess.tokenizer.word_index) + 1
        print(self.input_shape, self.output_sequence_length, self.french_vocab_size)

        self.en_data = en_data
        self.fr_data = fr_data

    def build_model(self):
        self.model = Sequential()
        self.model.add(GRU(self.output_sequence_length,
                           input_shape=self.input_shape,
                           return_sequences=True
                           )
                       )
        self.model.add(TimeDistributed(Dense(self.french_vocab_size)))
        self.model.add(Activation("softmax"))

        self.model.compile(**self.compile_args)

    def fit_model(self):
        self.model.fit(self.en_data, self.fr_data, **self.fit_args)
