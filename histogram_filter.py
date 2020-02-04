import numpy as np
import inspect

class HistogramFilter(object):
    """
    Class HistogramFilter implements the Bayes Filter on a discretized grid space.
    """

    def histogram_filter(self, cmap, belief, action, observation):
        '''
        Takes in a prior belief distribution, a colormap, action, and observation, and returns the posterior
        belief distribution according to the Bayes Filter.
        :param cmap: The binary NxM colormap known to the robot.
        :param belief: An NxM numpy ndarray representing the prior belief.
        :param action: The action as a numpy ndarray. [(1, 0), (-1, 0), (0, 1), (0, -1)]
        :param observation: The observation from the color sensor. [0 or 1].
        :return: The posterior distribution.
        Use starter.npz data given in /data to debug and test your code. starter.npz has sequence of actions and 
        corresponding observations. True belief is also given to compare your filters results with actual results. 
        cmap = arr_0, actions = arr_1, observations = arr_2, true state = arr_3
    
        ### Your Algorithm goes Below.
        '''

        cmap = np.rot90(cmap, 1)
        belief = np.rot90(belief, 1)
        # ACTION UPDATE
        
        b = np.copy(belief)   # moves, multiplied my 0.9
        a = np.copy(belief)   # stationary, multiplied by 0.1
        
        a = a * 0.1
        b = b * 0.9
        
        #print('action: ', action)
        
        # right
        if action[1] == 1:
            a[:, -1] *= 10
            b = np.roll(b, 1, axis = 1)
            b[:, 0] = 0
            
        # left
        elif action[1] == -1:
            a[:, 0] *= 10
            b = np.roll(b, -1, axis = 1)
            b[:, -1] = 0
            
        # down
        elif action[0] == 1:
            a[-1, :] *= 10
            b = np.roll(b, 1, axis = 0)
            b[0, :] = 0
            
        # up
        elif action[0] == -1:
            a[0, :] *= 10
            b = np.roll(b, -1, axis = 0)
            b[-1, :] = 0
            
        belief = a+b
        
                # SENSOR UPDATE
        
        color1 = cmap
        color0 = 1 - color1
        
        if observation == 1:
            res_color = (color1 * 0.9) + (color0 * 0.1)
            belief = belief * res_color
        elif observation == 0:
            res_color = (color1 * 0.1) + (color0 * 0.9)
            belief = belief * res_color      
            
        # NORMALISE
        
        tot = np.sum(belief)
        belief = belief/tot
            
        max_like_state = np.unravel_index(belief.argmax(), belief.shape)
        
        max_like_state = np.array(max_like_state)
         
        global local_vars
        local_vars = inspect.currentframe().f_locals
        
        
        cmap = np.rot90(cmap, 3)
        belief = np.rot90(belief, 3)
        
        return (max_like_state, belief, local_vars)
        
            
            
            
            