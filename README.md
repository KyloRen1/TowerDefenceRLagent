# Tower Defence RL Agent

<p align="center">
  <img src="assets/game.gif" alt="GIF 1" width="50%">
</p>

</br>

This project is an implementation of a classical [Tower Defence](https://www.youtube.com/watch?v=L8ypSXwyBds) game, which allows you to play it on your own, and it also includes a `gym`-like environment for training RL agents.

### Current progress

- [x] PyGame Tower Defence
- [x] Tower Defence as gym environment
- [ ] DQN - IN PROGRESS 
- [ ] inference script for DQN 
- [ ] Gameplay recording for Immitation learning



## Getting started

1. Clone repository
```python
git clone https://github.com/KyloRen1/TowerDefenceRLagent
```

2. Create python environment
```python
python3 -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

3. Run game
```python
python -m src.main_game --cfg configs/game_config.yml
```

4. Run agent training
```python
python -m src.main_agent --game-cfg configs/game_config.yml \
    --agent-cfg configs/agent_config.yml --world-speed 10
```