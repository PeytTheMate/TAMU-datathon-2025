# training/utils.py
import numpy as np
import torch

def state_to_tensor(state):
    board = np.zeros((10, 18, 20), dtype=np.float32)
    
    my_trail = state['agent1_trail']
    opp_trail = state['agent2_trail']
    my_head = my_trail[-1]
    opp_head = opp_trail[-1]

    for x, y in my_trail:
        board[0, y, x] = 1.0
    for x, y in opp_trail:
        board[1, y, x] = 1.0
    board[2, my_head[1], my_head[0]] = 1.0
    board[3, opp_head[1], opp_head[0]] = 1.0

    board[8, :, :] = state['agent1_boosts'] / 3.0
    board[9, :, :] = min(state['turn_count'] / 200.0, 1.0)
    
    return torch.from_numpy(board)