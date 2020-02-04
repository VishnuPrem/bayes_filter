import numpy as np
import matplotlib.pyplot as plt
from histogram_filter import HistogramFilter
import random

import inspect


if __name__ == "__main__":
    
    # Load the data
    data = np.load(open('data/starter.npz', 'rb'))
    cmap = data['arr_0']
    actions = data['arr_1']
    observations = data['arr_2']
    belief_states = data['arr_3']
    
    tot_time_steps = len(actions)
    (map_m, map_n) = cmap.shape
    bayes_filter = HistogramFilter()
    
    uniform_prob = 1 / (map_m * map_n * 1.0)
    prior_state = np.full((map_m, map_n), uniform_prob)
    
    for t in range(30):
        
        (max_like_state, prior_state, lv) = bayes_filter.histogram_filter(cmap, prior_state, actions[t], observations[t])
        print(max_like_state)
    
