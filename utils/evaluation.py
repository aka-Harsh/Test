import numpy as np

def calculate_probabilities(predictions):
    diff = np.diff(predictions)
    rise_count = np.sum(diff > 0)
    fall_count = np.sum(diff < 0)
    total_count = len(diff)
    
    rise_prob = (rise_count / total_count) * 100
    fall_prob = (fall_count / total_count) * 100
    
    return rise_prob, fall_prob
