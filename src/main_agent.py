import click
import pygame as pg

from src.game.utils import (
  load_config, 
  create_game_window, 
  display_data, 
  draw_text, 
  game_result_plot)
from src.game.world import World
from src.game.button import Button
from src.game.enemy import Enemy
from src.game.turret import create_turret, select_turret, clear_selection


@click.command(help="")
@click.option("--cfg", type=str, help="config file path")
def main(cfg):
    # actions: positions to place the turret; upgrade it
    # rewards: +1 for killed enemy, +100 for won game, -100 for lost game, -10 for lost life
    # state: image of the screen 

    pass

if __name__ == '__main__':
  main()