def get_poses_from_file(dataset_path):
	"""
	Load poses from monolithic file.
	"""
	print('Loading poses from monolithic file...')

	# For importing poses
	import sys
	import os
	sys.path.append(os.path.expanduser("/workspace/code/corelibs/src/tools-python"))
	sys.path.append(os.path.expanduser("/workspace/code/corelibs/build/datatypes"))
	sys.path.append(os.path.expanduser("/workspace/code/corelibs/build/datatypes/datatypes_python"))

	from mrg.logging import MonolithicDecoder
	from mrg.adaptors.transform import PbSerialisedTransformToPython
	# from mrg.transform.conversions import se3_to_components, build_se3_transform

	# Open monolithic and iterate frames
	RO_relative_poses_path = dataset_path+"ro_relative_poses.monolithic"
	print("reading RO_relative_poses_path: " + RO_relative_poses_path)
	monolithic_decoder = MonolithicDecoder(
	    RO_relative_poses_path)

	# iterate mono
	RO_se3s = []
	RO_timestamps = []
	for pb_serialised_transform, _, _ in monolithic_decoder:
	    # adapt
	    serialised_transform = PbSerialisedTransformToPython(
	        pb_serialised_transform)
	    RO_se3s.append(serialised_transform[0])
	    RO_timestamps.append(serialised_transform[1])
	print("Finished reading",len(RO_timestamps),"poses.")
	return RO_se3s, RO_timestamps