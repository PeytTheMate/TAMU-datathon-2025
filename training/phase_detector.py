# training/phase_detector.py
import networkx as nx

def is_separated(state, player):
    """Return True if agents are in separate regions (survival phase)"""
    G = nx.grid_2d_graph(20, 18, periodic=True)
    # Remove all walls/trails
    for y in range(18):
        for x in range(20):
            if state['board'][y][x] != 0:
                G.remove_node((x, y))
    
    my_head = state['agent1_trail'][-1] if player == 1 else state['agent2_trail'][-1]
    opp_head = state['agent2_trail'][-1] if player == 1 else state['agent1_trail'][-1]
    
    try:
        return not nx.has_path(G, my_head, opp_head)
    except:
        return True  # Graph disconnected