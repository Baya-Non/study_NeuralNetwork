import os.path
import gzip
import pickle
import os
import numpy as np
import get_mnist

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist_data/mnist.pkl"

def change_one_hot_label(X):
	# 一次元がXのsize数、二次元が10の、値が0の行列を作る
	T = np.zeros((X.size,10))

	# Tからkeyとvalueをそれぞれidxとrowとして取り出す。
	for idx, row in enumerate(T):
		# row内のXのidxに当たる値keyに対して、そのkeyに値1を代入して追加する。
		# 例えばidx = 3なら[0,0,1,0,0,0,0,0,0,0]となる
		row[X[idx]] = 1

	return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
	print("start reshape mnist data...")

	if not os.path.isfile(save_file):
		os.mkdir(save_file)

	with open(save_file, 'rb') as f:
		dataset = pickle.load(f)

	if normalize:
		# ndarrayの要素のデータ型を、別のデータ型(float32)にしたndarrayを生成する。要素を変更しても元のndarrayには影響しない
		# 0〜255としてデータを持つ各ピクセルを0〜1としてのデータとして整形する
		for r_size in ('train_img', 'test_img'):
			dataset[r_size] = dataset[r_size].astype(np.float32)
			dataset[r_size] /= 255.0 

	if one_hot_label:
		dataset['train_label'] = change_one_hot_label(dataset['train_label'])
		dataset['test_label'] = change_one_hot_label(dataset['test_label'])

	if not flatten:
		for key in ('train_img', 'test_img'):
			dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

	print("reshape mnist data !!!")
	return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 