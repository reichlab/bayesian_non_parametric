import numpy as np

def create_train_test(data,lag_num,break_point):
	X_train = []
	y_train = []
	X_test = []
	y_test = []

	train_data = data[:break_point]
	test_data = data[break_point:]

	for i in range(lag_num-1,len(train_data)-1):
		for j in range(i+1-lag_num,i+1):
			X_train.append(train_data[j])
		y_train.append(train_data[i+1])

	for i in range(lag_num-1,len(test_data)-1):
		for j in range(i+1-lag_num,i+1):
			X_test.append(test_data[j])
		y_test.append(test_data[i+1])
	X_train = np.array(X_train,dtype=np.float32).reshape((-1,lag_num))
	y_train = np.array(y_train,dtype=np.float32).reshape((-1,1))
	X_test = np.array(X_test,dtype=np.float32).reshape((-1,lag_num))
	y_test = np.array(y_test,dtype=np.float32).reshape((-1,1))

	return X_train,y_train,X_test,y_test