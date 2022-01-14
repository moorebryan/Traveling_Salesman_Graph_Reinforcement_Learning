''' This code defines the actual agent for our game
    following the same format as open gym ai.
    A typical agent in reinforcement learning models is a class 
    that has the ability to navigate the environment( in this case the NN),
    take an action and update the rewards from that action
'''    

import numpy as numpy
import torch
import random
import math
from collections import namedtuple
import os 
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Agent():
    def __init__(self, model, optimizer, lr_scheduler):
        self.model = model  # The actual QNet
        self.optimizer = optimizer # Choice of optimizer for instance Adam
        self.lr_scheduler = lr_scheduler # learning scheduler similar to how often updates are to be made
        self.loss_fn = nn.MSELoss()# loss function(m.s.e in this case)
    
    def predict(self, state_tsr, W):
        # batch of 1 --> only called at inference time
        # enabling no grad temporarily disables all gradient computations, in other words
            # to save computation time if we dont specify this autograd is enabled
            # and all gradient steps are set up to be calculated during the forward pass
        with torch.no_grad():
            # using unsqueeze will reshape the two tensors to have the batch size dimension added
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0] # gets list of rewards (uncouples list within list)
    
    def get_best_action(self, state_tsr, state):
        """ Computes the best (greedy) action to take from a given state
            Returns a tuple containing the ID of the next node and the corresponding estimated reward.
            Each time the starting state is different hence you get a different prediction
        """
        W = state.W # named tuples hence state.W gives us the (10,10) weight
        estimated_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)
        sorted_reward_idx = estimated_rewards.argsort(descending=True)# starting from this node visiting
        # sort indices of city that gave us best reward
        
        solution = state.partial_solution # list of all visited nodes so far
        
        already_in = set(solution)
        for idx in sorted_reward_idx.tolist():
            if (len(solution) == 0 or W[solution[-1], idx] > 0) and idx not in already_in:
                return idx, estimated_rewards[idx].item()
        
    def batch_update(self, states_tsrs, Ws, actions, targets):
        """ Take a gradient step using the loss computed on a batch of (states, Ws, actions, targets)
        
            states_tsrs: list of (single) state tensors
            Ws: list of W tensors
            actions: list of actions taken
            targets: list of targets (resulting estimated rewards after taking the actions)
        """  
        # stack along the 'O'th dimension, here is where we exploits the batch size dimension by
        # replaying the event for different starting positions of the graph. 
        # Each permutation generates a (state,ws action,target)
        Ws_tsr = torch.stack(Ws).to(device)
        xv = torch.stack(states_tsrs).to(device)

        ## PyTorch accumulates the gradients on subsequent backward passes. So make sure for each epoch
        ## we clear out the gradients from the last pass
        self.optimizer.zero_grad()
        
        # the rewards estimated by Q for the given actions
        estimated_rewards = self.model(xv, Ws_tsr)[range(len(actions)), actions]
        #below you update the loss through backward prop
        loss = self.loss_fn(estimated_rewards, torch.tensor(targets, device=device))
        loss_val = loss.item()
        
        loss.backward()
        self.optimizer.step()        
        self.lr_scheduler.step()
        
        return loss_val
