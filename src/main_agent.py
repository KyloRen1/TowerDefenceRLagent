import click
import time
import os

from src.game.tower_defence import TowerDefence
from src.game.utils import load_config
from src.agent.pipeline import Agent
from src.agent.utils import plot_scores

os.environ["TORCH_HOME"] = "./.cache"
os.environ["TORCH_EXTENSIONS_DIR"] = "./.cache"
os.environ["TRANSFORMERS_CACHE"] = "./.cache"

@click.command(help="")
@click.option("--game-cfg", type=str, help="game config file path")
@click.option("--agent-cfg", type=str, help="agent config file path")
@click.option("--world-speed", type=int, default=2, 
    help="speed of the enemies in the world")
def main(game_cfg, agent_cfg, world_speed):
    # actions: positions to place the turret; upgrade it
    # rewards: +1 for killed enemy, +100 for won game, -100 for lost game, -10 for lost life
    # state: image of the screen
    game_cfg = load_config(game_cfg)
    agent_cfg = load_config(agent_cfg)

    # initializing game
    game = TowerDefence(game_cfg)
    game.world.game_speed = world_speed

    # initializing agent
    agent = Agent(agent_cfg, 
        num_classes=(game_cfg.game.screen.rows, game_cfg.game.screen.cols))

    scores = list()
    mean_scores = list()
    total_score = 0 
    best_score = -1e10

    print('Starting training')
    while True:
        # get old state
        state_old = game.get_state()

        # get action
        action = agent.get_action(state_old)

        # game step
        reward, done, score = game.step(action)
        state_new = game.get_state()

        # train short memory
        agent.train_short_memory(state_old, action, reward, state_new, done)

        target_net_state_dict = agent.trainer.target_model.state_dict()
        policy_net_state_dict = agent.trainer.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (
                policy_net_state_dict[key] * agent_cfg.agent.tau + 
                target_net_state_dict[key] * (1 - agent_cfg.agent.tau)
            )

        # remember 
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            #agent.train_long_memory()

            agent.trainer.policy_model.save('latest_model.pth')
            if score >= best_score:
                best_score = score 
                agent.trainer.policy_model.save('best_model.pth')
                agent.trainer.policy_model.save(f'model_{agent.n_games}.pth')
            
            # agent statistics logs
            print(f'Game: {agent.n_games} | Score: {score} | Best score: {best_score}')

            # plotting
            scores.append(score)
            total_score += score 
            mean_score = total_score / agent.n_games 
            mean_scores.append(mean_score)
            plot_scores(scores, mean_scores)

            time.sleep(10)


if __name__ == '__main__':
    main()
