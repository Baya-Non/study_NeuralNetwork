# coding: utf-8
import sys, os
from commons.func import *
from commons.grad import numerical_grad
sys.path.append(os.pardir) 



class TwoLayerNetWork:
	
	#クラス初期化メゾット
	#imputsize = 784 初期ノード数
	#outputsize = 10 #出力ノード数
	#hidden_size = 100 隠れ中間ノード数

	def __init__(self, input_size, hidden_size,output_size, weight_init_std=0.01):	
		# ガウス分布に従う乱数で初期化
		# weight(重み)の初期化
		# バイアス値の初期化
		# paramsは[784 * 100]次元
		self.params = {}
		self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

	# 認識・推論を行う
	def predict(self, x):
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']

		#xと画像データの重みをとり、それぞれsigmoid関数とsoftmax関数に入力する
		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		return y

	# 損失率から学習パラメータの推定を行う
	# x:入力データ, t:教師データ
	def loss(self, x, t):
		y1 = self.predict(x)
		y2 = cross_entropy_error(y1, t)
		return y2

	# 正答率を求める
	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis = 1)
		t = np.argmax(t, axis = 1)

		accuracy = np.sum(y == t) / float(x.shape[0])
		return accuracy

	# x:入力データ, t:教師データ
	#重みに対するパラメーターを求める
	def numerical_gradient(self, x, t):
		loss_W = lambda W: self.loss(x, t)

		grads = {}
		grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
		grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
		grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
		grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

		return grads

	def gradient(self, x, t):
		W1, W2 = self.params['W1'], self.params['W2']
		b1, b2 = self.params['b1'], self.params['b2']
		grads = {}

		# バッチサイズを読み取る
		batch_num = x.shape[0]

		# forward
		a1 = np.dot(x, W1) + b1
		z1 = sigmoid(a1)
		a2 = np.dot(z1, W2) + b2
		y = softmax(a2)

		# backward
		dy = (y -t) /batch_num
		grads['W2'] = np.dot(z1.T, dy)
		grads['b2'] = np.sum(dy, axis=0)

		da1 = np.dot(dy, W2.T)
		dz1 = sigmoid_grad(a1) * da1
		grads['W1'] = np.dot(x.T, dz1)
		grads['b1'] = np.sum(dz1, axis=0)

		return grads
