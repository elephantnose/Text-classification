import sys
from collections import defaultdict

import jieba
import gensim
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


def build_embeddings_matrix(word_vec_model):
	# 初始化词向量矩阵
	embeddings_matrix = np.random.random((len(word_vec_model.wv.vocab)+1, 128))
	# 初始化词索引字典
	word_index = defaultdict(dict)

	for index, word in enumerate(word_vec_model.index2word):
		word_index[word] = index + 1
		# 预留0行给查不到的词
		embeddings_matrix[index+1] = word_vec_model.get_vector(word)
	return word_index, embeddings_matrix


def build_model(word_index, embeddings_matrix):
	model = keras.Sequential()
	model.add(keras.layers.Embedding(input_dim=len(word_index)+1, 
									output_dim=128, 
									weights=[embeddings_matrix],
									input_length=20,
									trainable=False))
	model.add(keras.layers.GlobalAveragePooling1D())
	model.add(keras.layers.Dense(32, activation=tf.nn.relu))
	model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
	
	model.compile(optimizer=tf.train.AdamOptimizer(),
					loss='binary_crossentropy',
					metrics=['accuracy'])
	model.summary()
	return model


def	train_data(word_index):
	df = pd.read_csv("./waimai.csv", names=["label", "review"])
	df["word_index"] = df["review"].astype("str").map(lambda x: np.array([word_index[i] for i in x.split(" ")]))
	# 填充及截断
	train = keras.preprocessing.sequence.pad_sequences(df["word_index"].values, maxlen=20, padding='post', truncating='post', dtype="float32")
	x_train, x_test, y_train, y_test = train_test_split(train, df["label"].values, test_size=0.2, random_state=1)
	# 从训练集上分出验证集
	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)
	return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == '__main__':
	# 加载词向量
	word_vec_model = gensim.models.KeyedVectors.load_word2vec_format("word_vec.txt", binary=False)
	# 建立词索引
	word_index, embeddings_matrix = build_embeddings_matrix(word_vec_model)
	# 生成训练数据及测试数据, 验证集
	x_train, x_val, x_test, y_train, y_val, y_test = train_data(word_index)
	# 构建模型
	model = build_model(word_index, embeddings_matrix)
	# 训练
	model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
	# 评估
	results = model.evaluate(x_test, y_test)
	print(f"损失: {results[0]}, 准确率: {results[1]}")
	# 模型保存
	model.save_weights('./model/waimai_model')
