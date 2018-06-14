import os


def info(inf):
    def fu(func):
        print(inf)
        return func
    return fu


def load(path):
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')


@info("loading small_vocab_en")
def load_en():
    return load("data/small_vocab_en")


@info("loading small_vocab_fr")
def load_fr():
    return load("data/small_vocab_fr")
