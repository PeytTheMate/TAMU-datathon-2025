# Case Closed Hackathon Agent

## Overview
This repository implements a competitive agent for the TAMU 2025 Dathathon Case Closed Competition. The service exposes a lightweight Flask API that mirrors the tournament harness, while the core agent logic evaluates safe moves on the 18×20 toroidal arena by measuring accessible space and prioritising survival.

Use this README as the canonical guide for installing dependencies, running the HTTP service, exercising the agent locally, and preparing the Docker image for submission.

## Repository layout
- `agent.py` – Flask application plus decision logic for turn-by-turn move selection.
- `game/` – Local simulator for automated scrimmages (`case_closed_game.py` and helpers).
- `training/` – Optional reinforcement-learning utilities (Gymnasium-compatible environment).
- `eval/` – Scripts for analysing match logs and benchmarking variants.
- `Dockerfile` – CPU-only build recipe aligned with the event’s 5 GB limit.
- `requirements.txt` – Runtime dependencies used by both local execution and the container build.

## Prerequisites
- Python 3.11+
- pip (or uv/poetry as preferred, but pip examples are shown below), though uv is goated.
- Docker (for the final submission image)

## Install dependencies
```bash
pip install -r requirements.txt
```
This installs only Flask and NetworkX, keeping the runtime footprint.

## Run the agent service
```bash
python agent.py
```
The Flask server listens on **port 5008** and exposes:
- `GET /` – Metadata about the bot (useful for quick health checks).
- `POST /send-state` – Ingest the latest game state JSON payload.
- `GET /send-move` – Returns the selected direction (and optional `:BOOST`).
- `POST /end` – Acknowledge match completion.

Use standard tooling (e.g., `curl`, Postman) to exercise the API manually:
```bash
curl -X POST http://localhost:5008/send-state \
  -H "Content-Type: application/json" \
  -d '{
        "width": 20,
        "height": 18,
        "walls": [],
        "agent1_trail": [[1,1],[2,1],[3,1]],
        "agent2_trail": [[10,10],[10,11]],
        "agent1_boosts": 3,
        "agent2_boosts": 2,
        "player_number": 1
      }'

curl http://localhost:5008/send-move
```

## Automated scrimmages
With the service running locally (bare metal or Docker), launch automated matches via the built-in simulator:

```bash
pip install requests

python scripts/scrimmage.py
```

Create `scripts/scrimmage.py` with the following contents to pit the HTTP agent against a heuristic opponent for 20 games:
```python
import random
import requests
from game.case_closed_game import Game, Direction, GameResult

BASE_URL = "http://localhost:5008"

DIR_TO_NAME = {
    Direction.UP.value: "UP",
    Direction.DOWN.value: "DOWN",
    Direction.LEFT.value: "LEFT",
    Direction.RIGHT.value: "RIGHT",
}

RIGHT_TURN = {
    Direction.UP: Direction.RIGHT,
    Direction.RIGHT: Direction.DOWN,
    Direction.DOWN: Direction.LEFT,
    Direction.LEFT: Direction.UP,
}

def encode_state(game: Game) -> dict:
    return {
        "width": game.board.width,
        "height": game.board.height,
        "walls": [],
        "agent1_trail": game.agent1.get_trail_positions(),
        "agent2_trail": game.agent2.get_trail_positions(),
        "agent1_boosts": game.agent1.boosts_remaining,
        "agent2_boosts": game.agent2.boosts_remaining,
        "player_number": 1,
    }


def get_agent_move(state: dict):
    requests.post(f"{BASE_URL}/send-state", json=state, timeout=2)
    move_json = requests.get(f"{BASE_URL}/send-move", timeout=2).json()
    move_text = move_json["move"]
    use_boost = move_text.endswith(":BOOST")
    if use_boost:
        move_text = move_text.replace(":BOOST", "")
    direction = next(
        d for d in Direction if DIR_TO_NAME[d.value] == move_text
    )
    return direction, use_boost


def opponent_move(game: Game) -> Direction:
    current = game.agent2.direction
    preferred = RIGHT_TURN[current]
    for option in [preferred, current, RIGHT_TURN[preferred], RIGHT_TURN[RIGHT_TURN[current]]]:
        head = game.agent2.trail[-1]
        dx, dy = option.value
        nx, ny = (head[0] + dx) % game.board.width, (head[1] + dy) % game.board.height
        if (nx, ny) not in game.agent2.trail and (nx, ny) not in game.agent1.trail:
            return option
    return random.choice(list(Direction))


results = {GameResult.AGENT1_WIN: 0, GameResult.AGENT2_WIN: 0, GameResult.DRAW: 0}

for _ in range(20):
    game = Game()
    requests.post(f"{BASE_URL}/send-state", json=encode_state(game), timeout=2)
    while True:
        state = encode_state(game)
        agent_dir, agent_boost = get_agent_move(state)
        opp_dir = opponent_move(game)
        outcome = game.step(agent_dir, opp_dir, boost1=agent_boost, boost2=False)
        if outcome is not None:
            results[outcome] += 1
            requests.post(f"{BASE_URL}/end", timeout=2)
            break

print("Final tally:", {k.name: v for k, v in results.items()})
```

Interpret the output to gauge win/draw/loss counts for your bot (agent 1).

## Run automated checks
Before packaging or submitting, run static validation to ensure syntax sanity:
```bash
python -m compileall agent.py
```
Extend this section with unit tests or linting as you build more complex strategies.

## Build and test the Docker image
The competition infrastructure builds from the provided Dockerfile. Verify locally:
```bash
docker build -t case-closed-agent .
docker run -p 5008:5008 case-closed-agent
```
Then repeat the manual or automated scrimmage steps against `localhost:5008` to confirm parity with the bare-metal run.

With this workflow you should be able to iterate on strategies.
