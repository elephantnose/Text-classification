import pandas as pd
import jieba


def load_stop_words():
	"""加载停用词"""
	with open("../data_set/stop_words") as fr:
		stop_words = set([word.strip() for word in fr])
	return stop_words


if __name__ == '__main__':
	# 加载停用词
	stop_words = load_stop_words()
	# 读取文件
	df = pd.read_csv("../data_set/waimai_10k.csv")
	# 切词并过滤调停用词
	df["review"] = df["review"].map(lambda x: " ".join([i for i in jieba.cut(x) if i not in stop_words]))
	# 保存处理好的文本
	df.to_csv("./waimai.csv", index=False, header=False, columns=["label", "review"])

