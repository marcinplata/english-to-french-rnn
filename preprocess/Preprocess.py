import collections
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Preprocess:
    sentences = None
    counter = None
    tokenized_sentences = None
    tokenizer = None
    padded_sequences = None

    def __init__(self, sentences):
        self.sentences = sentences

    def get_counter(self):
        self.counter = collections.Counter([word for sentence in self.sentences for word in sentence.split()])
        return self.counter

    def print_most_common_words(self, number=10):
        if self.counter is None:
            self.get_counter()
        print("".join(str(c[1]) + "\t" + c[0] + "\n" for c in self.counter.most_common(number)))

    def tokenize(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.sentences)
        self.tokenized_sentences = self.tokenizer.texts_to_sequences(self.sentences)
        return self.tokenized_sentences, self.tokenizer

    def pad(self, length=None):
        if self.tokenized_sentences is None:
            self.tokenize()
        self.padded_sequences = pad_sequences(self.tokenized_sentences, maxlen=length, padding='post')
        return self.padded_sequences

    def preprocess(self):
        self.tokenize()
        self.pad()
        return self.padded_sequences, self.tokenizer

    def preprocess_with_params(self, tokenizer, length):
        self.tokenized_sentences = tokenizer.texts_to_sequences(self.sentences)
        self.pad(length)
        return self.padded_sequences, tokenizer

    def get_preprocessed_sequences(self):
        if self.padded_sequences is None:
            self.preprocess()
        return self.padded_sequences
