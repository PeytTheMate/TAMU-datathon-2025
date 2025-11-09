"""HTTP agent entrypoint for the Case Closed Tron-style competition."""

import os
import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from flask import Flask, jsonify, request


PARTICIPANT = "Peyton"
AGENT_NAME = "a1k0n-Stolen-94%"


Coord = Tuple[int, int]

app = Flask(__name__)
game_state: Dict[str, object] = {}


class Tron:
    """Maintains the current arena state and chooses moves for our agent."""

    _DIRS: Dict[str, Coord] = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0),
    }

    def __init__(self) -> None:
        self.width = 20
        self.height = 18
        self.me = 1
        self.them = 2
        self.board: List[List[int]] = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.my_pos: Optional[Coord] = None
        self.their_pos: Optional[Coord] = None
        self.my_dir: Optional[Coord] = None

    def update_state(self, state: Dict) -> None:
        """Refresh the cached board representation from the game server payload."""

        width = state.get("width", self.width)
        height = state.get("height", self.height)
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]

        self._clear_board()

        for wall in state.get("walls", []):
            self._mark_cell(wall, -1)

        for idx in (1, 2):
            for cell in state.get(f"agent{idx}_trail", []):
                self._mark_cell(cell, idx)

        player = state.get("player_number", 1)
        self.me = player
        self.them = 1 if player == 2 else 2

        my_trail = state.get(f"agent{player}_trail", [])
        their_trail = state.get(f"agent{self.them}_trail", [])

        self.my_pos = tuple(my_trail[-1]) if my_trail else None
        self.their_pos = tuple(their_trail[-1]) if their_trail else None

        if len(my_trail) > 1:
            prev = my_trail[-2]
            self.my_dir = (self.my_pos[0] - prev[0], self.my_pos[1] - prev[1])  # type: ignore[index]
        else:
            self.my_dir = None

    def get_move(self) -> str:
        """Choose the direction that maximizes our reachable area advantage."""

        if not self.my_pos:
            return "UP"

        scored_moves: List[Tuple[float, str]] = []
        fallback_moves: List[str] = []

        for name, vector in self._DIRS.items():
            nx = self.my_pos[0] + vector[0]
            ny = self.my_pos[1] + vector[1]

            if not self._in_bounds(nx, ny):
                continue

            fallback_moves.append(name)

            if self.board[ny][nx] != 0:
                continue

            score = self._evaluate_move((nx, ny))
            scored_moves.append((score, name))

        if scored_moves:
            scored_moves.sort(reverse=True)
            return scored_moves[0][1]

        # If every option is blocked we still need to return *something* predictable.
        return fallback_moves[0] if fallback_moves else "UP"

    def flood_fill(
        self,
        start_x: int,
        start_y: int,
        board: Optional[List[List[int]]] = None,
        allow: Optional[Iterable[Coord]] = None,
    ) -> int:
        """Count accessible cells from a starting point with simple BFS."""

        if not self._in_bounds(start_x, start_y):
            return 0

        board_view = board if board is not None else self.board
        allowed: Set[Coord] = set(allow or [])
        allowed.add((start_x, start_y))

        visited = [[False for _ in range(self.width)] for _ in range(self.height)]
        queue: deque[Coord] = deque([(start_x, start_y)])
        visited[start_y][start_x] = True
        reachable = 0

        while queue:
            cx, cy = queue.popleft()
            reachable += 1

            for dx, dy in self._DIRS.values():
                nx = cx + dx
                ny = cy + dy

                if not self._in_bounds(nx, ny) or visited[ny][nx]:
                    continue

                if board_view[ny][nx] != 0 and (nx, ny) not in allowed:
                    continue

                visited[ny][nx] = True
                queue.append((nx, ny))

        return reachable

    def _evaluate_move(self, next_pos: Coord) -> float:
        """Score a move by comparing reachable area to our opponent."""

        board_after_move = self._copy_board()
        nx, ny = next_pos
        board_after_move[ny][nx] = self.me

        my_area = self.flood_fill(nx, ny, board_after_move)

        opponent_area = 0
        if self.their_pos:
            opp_board = self._copy_specific_board(board_after_move)
            tx, ty = self.their_pos
            opp_board[ty][tx] = 0
            opponent_area = self.flood_fill(tx, ty, opp_board)

        # Prefer continuing straight slightly when tied to reduce oscillations.
        direction_bonus = 0.0
        if self.my_dir:
            intended = (nx - self.my_pos[0], ny - self.my_pos[1])  # type: ignore[index]
            if intended == self.my_dir:
                direction_bonus = 0.1

        return my_area - opponent_area + direction_bonus

    def _clear_board(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.board[y][x] = 0

    def _copy_board(self) -> List[List[int]]:
        return [row[:] for row in self.board]

    @staticmethod
    def _copy_specific_board(board: Sequence[Sequence[int]]) -> List[List[int]]:
        return [list(row) for row in board]

    def _mark_cell(self, coord: Sequence[int], value: int) -> None:
        if len(coord) != 2:
            return
        x, y = coord
        if self._in_bounds(x, y):
            self.board[y][x] = value

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height


tron = Tron()

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
