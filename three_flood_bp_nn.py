# -- encoding: utf-8 --
# Author zcy
# Date 2017.03.10

from numpy import *
def sigmoid(x, d = False):
	if d == True:
		return x * (1 - x)
	else:
		return 1 / (1 + exp(-x))

'''训练神经网络，得到最终的权值和阈值
参数
X：训练数据集，是一个m行n列的numpy数组
Y：训练标签集，是一个m行1列的numpy数组
h：隐层神经元的个数

返回值
隐层和输出层的权值、阈值，类型为numpy数组
'''

def MyBP(X, Y, h):
	if shape(X)[0] != shape(Y)[0]:
		print("The line of X and Y must be same!")

	m = shape(X)[0]; n = shape(X)[1]; l = shape(Y)[1]

	#输入层-隐层权重
	whj = random.random((n, h))
	#隐层-输出层权重
	vih = random.random((h, l))
	#隐层阀值
	rh = random.random((1, h))
	#输出层阀值
	thetaj = random.random((1, l))

	for i in range(50000):
		#正向传播
		#隐层输入
		alphah = dot(X, whj)
		#隐层输入
		bh = sigmoid(alphah - rh)
		#输出层输入
		betaj = dot(bh, vih)
		#输出层输出
		ykj = sigmoid(betaj - thetaj)
		#误差
		E = Y - ykj

		#反向传播
		gj = sigmoid(ykj, True) * E
		#隐层-输出层权重改变
		delta_vih = dot(bh.T, gj)

		#输出层阀值改变，由所有样本的平均值修正
		delta_thetaj1 = -gj
		for i in range(1, m):
			delta_thetaj1[0, :] += delta_thetaj1[i, :]
		delta_thetaj = delta_thetaj1[0, :] / m

		#输入层-隐层权重改变
		delta_whj = dot(X.T, ((sigmoid(bh, True)) * dot(gj, vih.T)))

		#隐层阀值改变，由所有样本的平均值修正
		delta_rh1 = (sigmoid(bh, True)) * dot(gj, vih.T)
		for i in range(1, m):
			delta_rh1[0, :] += delta_rh1[i, :]
		delta_rh = delta_rh1[0, :] / m

		#修正
		vih += delta_vih
		thetaj += delta_thetaj
		whj += delta_whj
		rh += delta_rh

	return whj, rh, vih, thetaj

'''通过权值和阈值预测新样本的类别
参数
X：训练数据集，是一个m行n列的numpy数组
Y：训练标签集，是一个m行1列的numpy数组
NewX：新样本数据集，是一个x行n列的numpy数组
h：隐层神经元的个数

返回值
新样本的类别，类型为numpy数组
'''
def predictY(X, Y, NewX, h):
	whj, rh, vih, thetaj = MyBP(X, Y, h)
	alphah = dot(NewX, whj)
	# python中数组同列不同行可以加减，少行的用上一行补充
	bh = sigmoid(alphah - rh)
	betaj = dot(bh, vih)
	NewY = sigmoid(betaj - thetaj)
	return NewY