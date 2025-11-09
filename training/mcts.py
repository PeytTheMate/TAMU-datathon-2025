# training/mcts.py
import numpy as np
from training.utils import state_to_tensor

class MCTSNode:
    def __init__(self, prior=0.0):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children = {}

def mcts_search(state, model, player, simulations=80):
    root = MCTSNode()
    for _ in range(simulations):
        node = root
        current_state = dict(state)
        path = [node]

        # Selection
        while node.children:
            children = node.children
            best_score = -float('inf')
            best_action = None
            for action, child in children.items():
                Q = child.total_value / (child.visit_count + 1)
                U = 1.4 * child.prior * np.sqrt(node.visit_count + 1) / (1 + child.visit_count)
                score = Q + U
                if score > best_score:
                    best_score = score
                    best_action = action
            # Simulate move (skip invalid for speed)
            node = children[best_action]
            path.append(node)

        # Expansion & Evaluation
        obs = state_to_tensor(current_state, player).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = model(obs)
        policy = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
        value = value.item()

        # Only expand valid actions
        for a in range(8):
            if is_valid_action(current_state, player, a):
                node.children[a] = MCTSNode(prior=policy[a])

        # Backpropagation
        for n in reversed(path):
            n.visit_count += 1
            n.total_value += value

    if not root.children:
        return 3  # RIGHT fallback
    return max(root.children, key=lambda a: root.children[a].visit_count)

def is_valid_action(state, player, action):
    # Simplified: don't go opposite + boost check
    my_trail = state['agent1_trail'] if player == 1 else state['agent2_trail']
    if len(my_trail) < 2:
        return True
    prev = my_trail[-2]
    head = my_trail[-1]
    dx = head[0] - prev[0]
    dy = head[1] - prev[1]
    opposites = {(1,0): (-1,0), (-1,0): (1,0), (0,1): (0,-1), (0,-1): (0,1)}
    new_dx = (1, 0, -1, 0)[action % 4]
    new_dy = (0, 1, 0, -1)[action % 4]
    if (new_dx, new_dy) == opposites.get((dx, dy), None):
        return False
    if action >= 4 and state[f'agent{player}_boosts'] == 0:
        return False
    return True