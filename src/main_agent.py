import click
import pygame as pg

from src.game.tower_defence import TowerDefence
from src.game.utils import load_config
from src.agent.pipeline import Agent
from src.agent.utils import plot_scores


@click.command(help="")
@click.option("--game-cfg", type=str, help="game config file path")
@click.option("--agent-cfg", type=str, help="agent config file path")
def main(game_cfg, agent_cfg):
    # actions: positions to place the turret; upgrade it
    # rewards: +1 for killed enemy, +100 for won game, -100 for lost game, -10 for lost life
    # state: image of the screen
    game_cfg = load_config(game_cfg)
    agent_cfg = load_config(agent_cfg)

    game = TowerDefence(game_cfg)
    agent = Agent(agent_cfg, 
        num_classes=(game_cfg.game.screen.rows, game_cfg.game.screen.cols))

    scores = list()
    mean_scores = list()
    total_score = 0 
    best_score = 0

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

        # remember 
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # train long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                best_score = score 
                agent.model.save()
            
            # agent statistics logs
            print(f'Game: {agent.n_games} | Score: {score} | Best score: {best_score}')

            # plotting
            scores.append(score)
            total_score += score 
            mean_score = total_score / agent.n_games 
            mean_scores.append(mean_score)
            plot_scores(scores, mean_scores)




if __name__ == '__main__':
    main()
