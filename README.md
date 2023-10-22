# Tower Defence RL Agent


![](assets/gam.gif)


This project is an implementation of [Tower Defence](https://www.youtube.com/watch?v=L8ypSXwyBds) game in classical setting where you can play it by your own and a `gym`-like environment version to train RL agents. 

#### Current progress

- [x] PyGame Tower Defence
- [x] Tower Defence as gym environment
- [ ] DQN - IN PROGRESS 
- [ ] inference script for DQN 
- [ ] Gameplay recording for Immitation learning



## Getting started

1. Clone repository
```python
https://github.com/KyloRen1/TowerDefenceRLagent
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