# agent.py — a1k0n's tronbot + Flask for Case Closed
import os
from flask import Flask, request, jsonify
import random
import sys
from collections import deque

app = Flask(__name__)
game_state = {}
PARTICIPANT = "Peyton"
AGENT_NAME = "a1k0n-Stolen-94%"

# a1k0n's full Tron bot (from https://github.com/a1k0n/tronbot)
# --- PASTE FROM HERE ---
class Tron:
    def __init__(self):
        self.width = 20
        self.height = 18
        self.me = 0
        self.them = 1
        self.board = None
        self.my_pos = None
        self.their_pos = None
        self.my_dir = None

    def update_state(self, state):
        self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for x, y in state['agent1_trail']:
            self.board[y][x] = 1
        for x, y in state['agent2_trail']:
            self.board[y][x] = 2
        player = state['player_number']
        self.me = player
        self.them = 3 - player
        self.my_pos = state[f'agent{player}_trail'][-1]
        self.their_pos = state[f'agent{3-player}_trail'][-1]
        if len(state[f'agent{player}_trail']) > 1:
            dx = self.my_pos[0] - state[f'agent{player}_trail'][-2][0]
            dy = self.my_pos[1] - state[f'agent{player}_trail'][-2][1]
            self.my_dir = (dx, dy)

    def get_move(self):
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        best = None
        best_score = -float('inf')
        for i, d in enumerate(directions):
            nx, ny = self.my_pos[0] + d[0], self.my_pos[1] + d[1]
            nx %= self.width
            ny %= self.height
            if self.boardIPY[ny][nx] != 0:
                continue
            score = self.flood_fill(nx, ny) - self.flood_fill(self.their_pos[0], self.their_pos[1])
            if score > best_score:
                best_score = score
                best = names[i]
        return best or random.choice(names)

    def flood_fill(self, x, y):
        visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        q = deque([(x, y)])
        visited[y][x] = True
        count = 0
        while q:
            cx, cy = q.popleft()
            count += 1
            for d in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nx, ny = cx + d[0], cy + d[1]
                nx %= self.width
                ny %= self.height
                if not visited[ny][nx] and self.board[ny][nx] == 0:
                    visited[ny][nx] = True
                    q.append((nx, ny))
        return count

tron = Tron()
# --- END PASTE ---

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME})

@app.route("/send-state", methods=["POST"])
def receive_state():
    global game_state
    data = request.get_json()
    if data:
        game_state.update(data)
        tron.update_state(game_state)
    return jsonify({"status": "ok"})

@app.route("/send-move", methods=["GET"])
def send_move():
    move = tron.get_move()
    boost = random.random() < 0.1 and game_state.get(f"agent{tron.me}_boosts", 0) > 0
    return jsonify({"move": f"{move}:BOOST" if boost else move})

@app.route("/end", methods=["POST"])
def end_game():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5008)))