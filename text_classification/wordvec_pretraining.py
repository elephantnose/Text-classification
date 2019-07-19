import pandas as pd
import gensim


if __name__ == '__main__':
	df = pd.read_csv("./waimai.csv", header=None)
	sentences = df.iloc[:, 1].astype("str").map(lambda x: x.split(" "))
	model = gensim.models.Word2Vec(sentences, size=128, workers=4, min_count=0)
	model.wv.save_word2vec_format('./word_vec.txt', binary=False)

