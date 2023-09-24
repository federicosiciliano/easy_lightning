import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.utils import resample


def bootstrap_accuracy(y,pred_y,n_iters=100):
	app = []
	for i in range(n_iters):
		sub_y, sub_pred_y = resample(y,pred_y)
		app.append(accuracy_score(sub_y,sub_pred_y))
		#print(app[-1])
	return (np.mean(app),np.std(app))

def bootstrap_mse(y,pred_y,n_iters=100):
	app = []
	for i in range(n_iters):
		sub_y, sub_pred_y = resample(y,pred_y)
		app.append(mean_squared_error(sub_y,sub_pred_y))
		#print(app[-1])
	return (np.mean(app),np.std(app))

'''
def bootstrap_ndcg(y,pred_y,n_iters=100):
	app = []
	for i in range(n_iters):
		sub_y, sub_pred_y = resample(y,pred_y)
		app.append(custom_ndcg(sub_y,sub_pred_y))
		#print(app[-1])
	return (np.mean(app),np.std(app))
'''