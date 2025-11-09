# agent.py
import os, torch, numpy as np, networkx as nx
from flask import Flask, request, jsonify
from training.utils import state_to_tensor
from training.phase_detector import is_separated
from training.mcts import mcts_search

device = torch.device("cpu")
model = torch.load("models/interactive_final.pt", map_location=device)
model.eval()

app = Flask(__name__)
game_state = {}
DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": "Peyton", "agent_name": "MRL-TronSOTA"})

@app.route("/send-state", methods=["POST"])
def receive_state():
    global game_state
    game_state.update(request.get_json())
    return jsonify({"status": "ok"})

@app.route("/send-move", methods=["GET"])
def send_move():
    player = int(request.args.get("player_number", 1))
    obs = state_to_tensor(game_state, player).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(obs)
        policy, _ = model.forward_heads(features)
        action = np.argmax(policy.numpy())

    dir_str = DIRECTIONS[action % 4]
    boost = action >= 4 and game_state.get(f"agent{player}_boosts", 0) > 0
    move = f"{dir_str}:BOOST" if boost else dir_str
    return jsonify({"move": move})

@app.route("/end", methods=["POST"])
def end(): return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5008)