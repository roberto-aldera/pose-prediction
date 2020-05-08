import numpy as np

def get_bin_idx_and_delta(num_bins,min_bin,max_bin,raw_val):
    bin_range = (max_bin - min_bin)/num_bins
    bin_values = []
    for i in range(num_bins):
        bin_values.append(min_bin + i*bin_range + bin_range/2) # Bin values are to be in the middle of the bin
    # print("Bin range:",bin_range)
    # print("Bin centres:",bin_values)

    bin_index = np.int(np.floor((raw_val - min_bin)/bin_range))
    delta = np.float32(np.mod(raw_val,bin_range) - bin_range/2)

    if(bin_index >= num_bins):
        violation_distance = bin_index - (num_bins-1)
        bin_index = num_bins-1
        delta = delta + violation_distance * bin_range
    
    if(bin_index < 0):
        violation_distance = np.abs(bin_index)
        bin_index = 0
        delta = delta - violation_distance * bin_range

#     print("Bin value:",bin_values[int(bin_index)])
    return bin_index, delta

def get_value_from_bin_idx_and_delta(num_bins,min_bin,max_bin,bin_index,delta):
    bin_range = (max_bin - min_bin)/num_bins
    bin_values = []
    for i in range(num_bins):
        bin_values.append(min_bin + i*bin_range + bin_range/2) # Bin values are to be in the middle of the bin
    # print("Bin range:",bin_range)
    # print("Bin centres:",bin_values)
    
    value = np.float32(bin_values[bin_index] + delta)
    return value