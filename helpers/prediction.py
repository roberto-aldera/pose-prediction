def appender(input_seq,new_val):
    """
    Append value to input sequence (treat as arrays)
    """
    import numpy as np

    input_seq = np.delete(input_seq,0)
    updated_input_seq = np.append(input_seq,new_val)
    return updated_input_seq
    # my_arr = np.array([0,2,4,6])
    # print(my_arr)
    # my_new_arr = appender(my_arr,8)
    # print(my_new_arr)

def get_predictions_out_to_horizon(model,horizon,input_seq,future_val):
    """
    Predict values out to a horizon
    """
    import numpy as np
    import torch
    from torch.autograd import Variable 

    cumulative_predictions = np.empty(0)
    cumulative_predictions = np.append(cumulative_predictions,future_val)
    for i in range(horizon-1):
        updated_input_seq = appender(input_seq,future_val)
        updated_input_seq = updated_input_seq[np.newaxis,...,np.newaxis]
        updated_input_seq = Variable(torch.tensor(updated_input_seq))
        next_prediction = model(updated_input_seq)

        # Update states
        input_seq = updated_input_seq
        future_val = next_prediction.data.numpy()+updated_input_seq[0][-1].data.numpy()
        cumulative_predictions = np.append(cumulative_predictions,future_val)
    return cumulative_predictions