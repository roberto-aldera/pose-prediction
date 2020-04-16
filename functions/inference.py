def query_at_index(model,dim1_data,dim2_data,dim3_data,num_frames,num_features,idx):
	"""
	Query network at a particular input index.
	"""
	import numpy as np
	import torch
	from torch.autograd import Variable 

	query_dim1 = []
	query_dim2 = []
	query_dim3 = []

	newest_frame_remaining = num_frames - 1
	for i in range(num_frames):
	    query_dim1.append(dim1_data[idx+newest_frame_remaining])
	    query_dim2.append(dim2_data[idx+newest_frame_remaining])
	    query_dim3.append(dim3_data[idx+newest_frame_remaining])
	    
	    newest_frame_remaining -= 1

	scaled_query_dim1 = (query_dim1 - dim1_data.mean())/dim1_data.std()
	scaled_query_dim2 = (query_dim2 - dim2_data.mean())/dim2_data.std()
	scaled_query_dim3 = (query_dim3 - dim3_data.mean())/dim3_data.std()

	# Build combination so x, y, and theta are adjacent for 1 pose
	c = np.empty(scaled_query_dim1.size + scaled_query_dim2.size + scaled_query_dim3.size)
	c[0::num_features] = scaled_query_dim1
	c[1::num_features] = scaled_query_dim2
	c[2::num_features] = scaled_query_dim3

	new_query = Variable(torch.Tensor([c]))

	answer = model(new_query)

	predicted_error_x = (answer[0][0]*dim1_data.std() + dim1_data.mean()).data
	predicted_error_y = (answer[0][1]*dim2_data.std() + dim2_data.mean()).data
	predicted_error_theta = (answer[0][2]*dim3_data.std() + dim3_data.mean()).data

	return predicted_error_x,predicted_error_y,predicted_error_theta