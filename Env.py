# Import routines

import numpy as np
import math
import random
from itertools import product
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + list(permutations([0,1,2,3,4], 2))
        self.state_space = list(product([0,1,2,3,4], np.arange(0,24), [0,1,2,3,4,5,6])) #(location, hour, day)
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [1 if i == state[0] else 0 for i in range(m)] + [1 if i == state[1] else 0 for i in range(t)] + [1 if i == state[2] else 0 for i in range(d)]

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        # but driver always has the option to go ‘offline’ so appending (0,0) action to available actions
        actions.append((0,0))
        # as we have appended (0,0) action to available actions, we need to append action-index for (0,0) to possible_actions_index
        possible_actions_index.append(0) # using in get_action() to choose action index
        
        return possible_actions_index,actions   



    def time_day_update_func(self, curr_hour, curr_day, time_spent):
        """updating the hour and day using time spent for the ride"""
        hour = curr_hour + int(time_spent)
        curr_hour = hour % 24
        days = hour // 24
        curr_day = (days+curr_day) % 7
        
        return curr_hour, curr_day


    
    def time_spent_func(self, state, action, Time_matrix):
        """calculating the time-spent for the ride(current location to pickup location and pickup location to drop location)"""
        curr_loc, curr_hour, curr_day = state[0], state[1], state[2]
        pickup_loc, drop_loc = action[0], action[1]
        if curr_loc == pickup_loc:
            time_curr_to_pick = 0
            pickup_loc = curr_loc
            # getting time taken from pickup location to drop location
            time_pick_to_drop = Time_matrix[pickup_loc][drop_loc][curr_hour][curr_day]
        else:
            # getting time taken from current location to pickup location
            time_curr_to_pick = Time_matrix[curr_loc][pickup_loc][curr_hour][curr_day]
            # calculating new time after reaching pickup location
            curr_hour, curr_day = self.time_day_update_func(curr_hour, curr_day, time_curr_to_pick)
            # getting time taken from pickup location to drop location
            time_pick_to_drop = Time_matrix[pickup_loc][drop_loc][curr_hour][curr_day]
        
        return time_curr_to_pick, time_pick_to_drop



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        if action == (0,0):
            reward = -C
        else:
            time_curr_to_pick, time_pick_to_drop = self.time_spent_func(state, action, Time_matrix)
            reward = R*time_pick_to_drop - C*(time_pick_to_drop + time_curr_to_pick)
            
        return reward



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        curr_loc, curr_hour, curr_day = state[0], state[1], state[2]
        pickup_loc, drop_loc = action[0], action[1]
        if action == (0,0):
            waiting_time = 1
            new_hour, new_day = self.time_day_update_func(curr_hour, curr_day, waiting_time)
            new_location = curr_loc
            next_state = (new_location, new_hour, new_day)
        else:
            time_curr_to_pick, time_pick_to_drop = self.time_spent_func(state, action, Time_matrix)
            # calculating new time after reaching pickup location
            curr_hour, curr_day = self.time_day_update_func(curr_hour, curr_day, time_curr_to_pick)            
            # calculating new time after reaching drop location
            new_hour, new_day = self.time_day_update_func(curr_hour, curr_day, time_pick_to_drop)
            new_location = drop_loc
            next_state = (new_location, new_hour, new_day)
            
        return next_state

    

    def step(self, state, action_idx, Time_matrix):
        """finding the next step by finding next_state, reward, total_time_spent"""
        action = self.action_space[action_idx]
        next_state = self.next_state_func(state, action, Time_matrix)
        reward = self.reward_func(state, action, Time_matrix)
        # calculating time spent
        if action == (0,0):
            total_time_spent = 1
        else:
            time_curr_to_pick, time_pick_to_drop = self.time_spent_func(state, action, Time_matrix)
            total_time_spent = time_curr_to_pick + time_pick_to_drop

        return (next_state, reward, total_time_spent)
            


    def reset(self):
        return self.action_space, self.state_space, self.state_init
