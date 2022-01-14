''' The actual training code using epsilon greedy
    below implements a version of n-step Q-learning, it checkpoints the best models (according to the median path length),
     and prints some information.
'''
import numpy as np
import torch
import random
import math
from collections import namedtuple
import os
import time

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from scipy.signal import medfilt
from Agent import Agent
import env
from Memory import Memory, Experience
from lib import total_distance, is_state_final, get_next_neighbor_random, get_graph_mat, plot_graph

SEED = 1  # A seed for the random number generator
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Train():
    def __init__(self, NR_NODES, EMBEDDING_DIMENSIONS, EMBEDDING_ITERATIONS_T, NR_EPISODES,
                 MEMORY_CAPACITY, N_STEP_QL, BATCH_SIZE, GAMMA, INIT_LR, LR_DECAY_RATE, MIN_EPSILON,
                 EPSILON_DECAY_RATE, FOLDER_NAME):
        self.NR_NODES = NR_NODES
        self.EMBEDDING_DIMENSIONS = EMBEDDING_DIMENSIONS
        self.EMBEDDING_ITERATIONS_T = EMBEDDING_ITERATIONS_T
        self.NR_EPISODES = NR_EPISODES
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.N_STEP_QL = N_STEP_QL
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.INIT_LR = INIT_LR
        self.LR_DECAY_RATE = LR_DECAY_RATE
        self.MIN_EPSILON = MIN_EPSILON
        self.EPSILON_DECAY_RATE = EPSILON_DECAY_RATE
        self.FOLDER_NAME = FOLDER_NAME

    def init_model(self, fname=None):
        """ Create a new model. If fname is defined, load the model from the specified file.
        """
        Q_net = env.QNet(self.EMBEDDING_DIMENSIONS, T=self.EMBEDDING_ITERATIONS_T).to(device)
        optimizer = optim.Adam(Q_net.parameters(), lr=self.INIT_LR)
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.LR_DECAY_RATE)
        
        if fname is not None:
            checkpoint = torch.load(fname)
            Q_net.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        Q_func = Agent(Q_net, optimizer, lr_scheduler)
        return Q_func, Q_net, optimizer, lr_scheduler

    def checkpoint_model(self, model, optimizer, lr_scheduler, loss, 
                        episode, avg_length):
        if not os.path.exists(self.FOLDER_NAME):
            os.makedirs(self.FOLDER_NAME)
        
        fname = os.path.join(self.FOLDER_NAME, 'ep_{}'.format(episode))
        fname += '_length_{}'.format(avg_length)
        fname += '.tar'
        
        torch.save({
            'episode': episode,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'loss': loss,
            'avg_length': avg_length
        }, fname)

    def run_training(self):
        # seed everything for reproducible results first:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        # Create module, optimizer, LR scheduler, and Q-function
        Q_func, Q_net, optimizer, lr_scheduler = self.init_model()

        # Create memory
        memory = Memory(self.MEMORY_CAPACITY)

        # Storing metrics about training:
        found_solutions = dict()  # episode --> (coords, W, solution)
        losses = []
        path_lengths = []

        # keep track of median path length for model checkpointing
        current_min_med_length = float('inf')
        coords= get_graph_mat(n=self.NR_NODES)

        for episode in range(self.NR_EPISODES):
            # sample a new random graph by reversing nodes
            # similar to Sim Anneal
            coords = get_graph_mat(n=self.NR_NODES)
            l = random.randint(2, len(coords) - 1)
            i = random.randint(0, len(coords) - l)
            coords[i : (i + l)]=reversed(coords[i : (i + l)])
            coords = np.asarray(coords)
            W_np = distance_matrix(coords, coords)
            W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)
            
            # current partial solution - a list of node index
            solution = [random.randint(0, self.NR_NODES-1)]
            
            # current state (tuple and tensor)
            current_state = env.State(partial_solution=solution, W=W, coords=coords)
            current_state_tsr = env.state2tens(current_state)
            
            # Keep track of some variables for insertion in replay memory:
            states = [current_state]
            states_tsrs = [current_state_tsr]  # we also keep the state tensors here (for efficiency)
            rewards = []
            actions = []
            
            # current value of epsilon
            epsilon = max(self.MIN_EPSILON, (1-self.EPSILON_DECAY_RATE)**episode)
            
            nr_explores = 0
            t = -1
            while not is_state_final(current_state):
                t += 1  # time step of this episode
                
                if epsilon >= random.random():
                    # explore
                    next_node = get_next_neighbor_random(current_state)
                    nr_explores += 1
                else:
                    # exploit
                    next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
                    if episode % 50 == 0:
                        print('Ep {} | current sol: {} / next est reward: {}'.format(episode, solution, est_reward))
                
                next_solution = solution + [next_node]
                
                # reward observed for taking this step . This is the actual ground truth reward      
                reward = -(total_distance(next_solution, W) - total_distance(solution, W))
                
                next_state = env.State(partial_solution=next_solution, W=W, coords=coords)
                next_state_tsr = env.state2tens(next_state)
                
                # store rewards and states obtained along this episode:
                states.append(next_state) # Note: this is list of state within states for a particular episode
                states_tsrs.append(next_state_tsr)
                rewards.append(reward)
                actions.append(next_node)
                
                # store our experience in memory, using n-step Q-learning:
                if len(solution) >= self.N_STEP_QL:
                    memory.remember(Experience(state=states[-self.N_STEP_QL],
                                            state_tsr=states_tsrs[-self.N_STEP_QL],
                                            action=actions[-self.N_STEP_QL],
                                            reward=sum(rewards[-self.N_STEP_QL:]),
                                            next_state=next_state,
                                            next_state_tsr=next_state_tsr))
                    
                if is_state_final(next_state):
                    for n in range(1, self.N_STEP_QL):
                        memory.remember(Experience(state=states[-n],
                                                state_tsr=states_tsrs[-n], 
                                                action=actions[-n], 
                                                reward=sum(rewards[-n:]), 
                                                next_state=next_state,
                                                next_state_tsr=next_state_tsr))
                
                # update state and current solution within the current graph episode
                current_state = next_state
                current_state_tsr = next_state_tsr
                solution = next_solution
                
                # take a gradient step
                # We stored a batch of states in our memory and will now use them to make an update.
                loss = None
                if len(memory) >= self.BATCH_SIZE and len(memory) >= 2000:
                    experiences = memory.sample_batch(self.BATCH_SIZE)
                    
                    batch_states_tsrs = [e.state_tsr for e in experiences]
                    batch_Ws = [e.state.W for e in experiences]
                    batch_actions = [e.action for e in experiences]
                    batch_targets = []
                    
                    for i, experience in enumerate(experiences):
                        target = experience.reward
                        if not is_state_final(experience.next_state):
                            _, best_reward = Q_func.get_best_action(experience.next_state_tsr, 
                                                                    experience.next_state)
                            target += self.GAMMA * best_reward
                        batch_targets.append(target)
                        
                    # print('batch targets: {}'.format(batch_targets))
                    loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
                    losses.append(loss)
                    
                    """ Save model when we reach a new low average path length
                    """
                    med_length = np.median(path_lengths[-100:])
                    if med_length < current_min_med_length:
                        current_min_med_length = med_length
                        self.checkpoint_model(Q_net, optimizer, lr_scheduler, loss, episode, med_length)
                        
            length = total_distance(solution, W)
            path_lengths.append(length)

            if episode % 10 == 0:
                print('Ep %d. Loss = %.3f / median length = %.3f / last = %.4f / epsilon = %.4f / lr = %.4f' % (
                    episode, (-1 if loss is None else loss), np.median(path_lengths[-50:]), length, epsilon,
                    Q_func.optimizer.param_groups[0]['lr']))
                found_solutions[episode] = (W.clone(), coords.copy(), [n for n in solution])

        # Here return the losses and path lengths
        return losses, path_lengths
