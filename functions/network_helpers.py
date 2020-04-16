import numpy as np
import torch
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.autograd import Variable 

# class Pose_Logistic(nn.Module):
# 	print(input_size)
# 	def __init__(self):
# 		super().__init__()
# 		self.lin = nn.Linear(input_size,output_size)

# 	def forward(self, xb):
# 		return self.lin(xb)

def loss_batch(model,loss_func,xb,yb,opt=None):
	loss=loss_func(model(xb),yb)

	if opt is not None:
		loss.backward()
		opt.step()
		opt.zero_grad()
		
	return loss.item(),len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
	validation_loss = []
	for epoch in range(epochs):
		model.train()
		for xb, yb in train_dl:
			loss_batch(model, loss_func, xb, yb, opt)

		model.eval()
		with torch.no_grad():
			losses, nums = zip(
				*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
			)
		val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

		validation_loss.append(val_loss)
		print(epoch, val_loss)
	return validation_loss

def get_data(train_ds, valid_ds, bs):
	return (
		DataLoader(train_ds, batch_size=bs, shuffle=True),
		DataLoader(valid_ds, batch_size=bs * 2),
	)

# def get_model():
# 	model = Pose_Logistic()
# 	return model, optim.SGD(model.parameters(), lr=lr)