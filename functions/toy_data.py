def make_toy_data(alpha=10,data_size=100):
    """
    Create toy data for debugging. Alpha value is used for scale.
    """
    import numpy as np

    # Creating the artificial dataset
    toy_data = np.arange(0,alpha,alpha/data_size)

    # adding noise
    for i in range(len(toy_data)):
        toy_data[i] += np.sin(i)*alpha/10

    toy_pred = []
    toy_model_err = []

    for i in range(len(toy_data)-1):
        toy_pred.append(toy_data[i+1] + (toy_data[i+1] - toy_data[i]))

    for i in range(len(toy_pred)-1):
        toy_model_err.append(toy_data[i+2] - toy_pred[i])
        
    return toy_model_err,toy_data,toy_pred