import numpy as np


def to_text(predictions, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return [' '.join([index_to_words[np.argmax(word)] for word in p]) for p in predictions]
