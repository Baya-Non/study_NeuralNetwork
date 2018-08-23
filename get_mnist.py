try:
	import urllib.request
except ImportError:
	raise ImportError('you should use python version 3.x')
import os.path
import os
import gzip
import pickle
import numpy as np

url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}


dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist_data/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

#フォルダー作るノォxおxおxおxおxおおおおおお
def make_dir():
	d_save_path = dataset_dir + "/mnist_data" 
	if os.path.isfile(d_save_path):
		os.mkdir(d_save_path)
	else:
		print("File already exists!")

def download(filename):
	file_path = dataset_dir + "/mnist_data/" + filename


	if os.path.exists(file_path):
		print("the file has exists")
		return

	print("Downloading " + filename + " ... ")
	#the mnist data save to file_path
	urllib.request.urlretrieve(url_base + filename, file_path)
	print("saved for" + file_path)
	print("Done!!")

def download_mnist():
	for i in key_file.values():
		print("key_file" + i)
		download(i)

def load_img(filename):
	file_path = dataset_dir + "/mnist_data/" + filename

	print("Converting " + filename + " to Numpy array ...")
	with gzip.open(file_path, 'rb') as f:
		data = np.frombuffer(f.read() ,np.uint8, offset=16)

	data = data.reshape(-1, img_size)
	print("Done!!")

	return data

def load_label(filename):
	file_path = dataset_dir + "/mnist_data/" + filename

	print("Converting " + filename + " to NumPy Array ...")
	with gzip.open(file_path, 'rb') as f:
		labels = np.frombuffer(f.read(), np.uint8, offset=8)

	print("Done!!")

	return labels


def convert_numpy():
    dataset = {}
    dataset['train_img'] =  load_img(key_file['train_img'])
    dataset['train_label'] = load_label(key_file['train_label'])    
    dataset['test_img'] = load_img(key_file['test_img'])
    dataset['test_label'] = load_label(key_file['test_label'])
    
    return dataset

if __name__ == '__main__':
	make_dir()
	download_mnist()
	dataset = convert_numpy()
	print("Creating pickle file ...")
	with open(save_file, 'wb') as f:
		pickle.dump(dataset, f, -1)
	print("Done!")
