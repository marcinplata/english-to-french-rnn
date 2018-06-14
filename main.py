from data.load import load_en, load_fr
from preprocess.Preprocess import Preprocess
from models.Simple import Simple
from postprocess.to_text import to_text

en_sentences = load_en()
fr_sentences = load_fr()

# preprocess
en_data = Preprocess(en_sentences)
en_text, en_text_tokenizer = en_data.preprocess()
print('head of english sentences set')
print(en_text[:5])
print('shape of english sentences set')
print(en_text.shape)

fr_data = Preprocess(fr_sentences)
fr_text, fr_text_tokenizer = fr_data.preprocess()
print('head of french sentences set')
print(fr_text[:5])
print('shape of french sentences set')
print(fr_text.shape)

# training
print("training")
simpleRNN = Simple(en_data, fr_data)
simpleRNN.build_model()
simpleRNN.summary()

# fit model and save
simpleRNN.fit_model()
simpleRNN.save_model_weights("model.md5")

# or load model
# simpleRNN.load_model_weights("model.md5")

# predictions (test of 5 head english sentences)
test_data = Preprocess(en_sentences[:5])
test_text, _ = test_data.preprocess_with_params(en_text_tokenizer, fr_text.shape[1])
test_text = test_text.reshape((*test_text.shape, 1))

predictions = simpleRNN.predict(test_text)
print("predictions")
print("".join(s + "\n" for s in to_text(predictions, fr_text_tokenizer)))
print("original")
print("".join(s + "\n" for s in fr_sentences[:5]))