import jieba
import gensim
import numpy as np
from tensorflow import keras

from train import build_model, build_embeddings_matrix
from text_preprocessing import load_stop_words


if __name__ == '__main__':
	word_vec_model = gensim.models.KeyedVectors.load_word2vec_format("word_vec.txt", binary=False)
	word_index, embeddings_matrix = build_embeddings_matrix(word_vec_model)
	model = build_model(word_index, embeddings_matrix)
	model.load_weights("./model/waimai_model")
	
	stop_words = load_stop_words()
	
	while True:
		text = input("请输入一句话：")
		text = [word_index.get(word, 0) for word in jieba.cut(text)]
		text = keras.preprocessing.sequence.pad_sequences([text], maxlen=20, padding='post', truncating='post', dtype="float32")

		res = model.predict(text)[0][0]
		if res >= 0.5:
			print(f"好评, 得分: {res*100}")
		else:
			print(f"差评，得分: {res*100}")

		print()

