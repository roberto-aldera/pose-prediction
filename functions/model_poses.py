def get_model_pose_prediction_and_error(RO_se3s):
	"""
	Build up array of model-predicted poses and the error from ground truth.
	"""
	import numpy as np

	pred_poses = []

	for i in range(len(RO_se3s)-1):
	    future_pose = RO_se3s[i+1]*(RO_se3s[i+1]*np.linalg.inv(RO_se3s[i]))
	    pred_poses.append(future_pose)

	dim1_data = np.zeros(len(pred_poses)-1)
	dim2_data = np.zeros(len(pred_poses)-1)
	dim3_data = np.zeros(len(pred_poses)-1)

	pred_poses_x = np.zeros(len(pred_poses))
	pred_poses_y = np.zeros(len(pred_poses))
	pred_poses_theta = np.zeros(len(pred_poses))

	for i in range(len(pred_poses)):
		pred_poses_x[i] = pred_poses[i][0,3]
		pred_poses_y[i] = pred_poses[i][1,3]
		pred_poses_theta[i] = np.arccos(pred_poses[i][0,0])

	for i in range(len(dim1_data)):
		dim1_data[i] = RO_se3s[i+2][0,3] - pred_poses[i][0,3]
		dim2_data[i] = RO_se3s[i+2][1,3] - pred_poses[i][1,3]
		dim3_data[i] = np.arccos(RO_se3s[i+2][0,0]) - np.arccos(pred_poses[i][0,0])

	return pred_poses,dim1_data,dim2_data,dim3_data,pred_poses_x,pred_poses_y,pred_poses_theta