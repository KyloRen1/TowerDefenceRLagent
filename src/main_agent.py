import click
import pygame as pg

from src.game.pipeline import TowerDefence

@click.command(help="")
@click.option("--game-cfg", type=str, help="game config file path")
@click.option("--agent-cfg", type=str, help="agent config file path")
def main(game_cfg, agent_cfg):
    # actions: positions to place the turret; upgrade it
    # rewards: +1 for killed enemy, +100 for won game, -100 for lost game, -10 for lost life
    # state: image of the screen 

    game = TowerDefence(game_cfg)
    
    pass

if __name__ == '__main__':
  main()