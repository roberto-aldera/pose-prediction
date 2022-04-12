def sliding_window_delta(data, seq_length):
    """
    Creates an x that contains input elements of length seq_length
    Generate corresponding y that stores the difference between the last
    element in x and the upcoming element
    """
    import numpy as np

    x = []
    y = []

    for i in range(len(data)-seq_length):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length] - data[i+seq_length-1]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

# my_arr = np.array([1,2,3,7,8,9,1,2,3])
# x,y = sliding_window_delta(my_arr,3)
# print(x)
# print(y)

def sliding_window_vanilla(data, seq_length, output_length):
    """
    Creates an x that contains input elements of length seq_length
    Generate corresponding y that stores the next n = output_length
    elements in the same larger sequence (so not using delta here)
    """
    import numpy as np

    x = []
    y = []

    for i in range(len(data)-seq_length-(1+output_length)):
        _x = data[i:(i+seq_length)]
        _y = data[(i+seq_length):(i+seq_length+output_length)]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)