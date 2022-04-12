def get_error_metrics(gt_predictions,model_predictions):
    """
    Find RMSE for short, medium, and long-term predictions
    """
    import numpy as np
    
    short_horizon = 5
    med_horizon = 20
    all_diffs = gt_predictions - model_predictions
    short_diffs = all_diffs[0:short_horizon]
    med_diffs = all_diffs[short_horizon:med_horizon]
    long_diffs = all_diffs[med_horizon:]
    short_rmse = np.sqrt(np.sum(np.square(short_diffs))/len(short_diffs))
    med_rmse = np.sqrt(np.sum(np.square(med_diffs))/len(med_diffs))
    long_rmse = np.sqrt(np.sum(np.square(long_diffs))/len(long_diffs))
    return(short_rmse, med_rmse, long_rmse)