import yaml
import pygame as pg
from typing import Tuple
from easydict import EasyDict
from src.game.world import World

# fonts for displaying text on the screen
text_font = None
large_font = None

# gui images 
coin_image = None
heart_image = None
logo_image = None


def load_config(cfg_path:str) -> EasyDict:
  """ 
    Reading config file 

  Args:
    cfg_path (str): path to the configuration file
  Returns:
    EasyDict: dictionary of configurations
  """
  with open(cfg_path, 'r') as f:
    config = EasyDict(yaml.safe_load(f))
  return config


def create_game_window(cfg:EasyDict) -> Tuple[pg.Surface, pg.time.Clock, pg.Surface]:
  global text_font, large_font
  global coin_image, heart_image, logo_image
  """ 
    Creating game window, game clock and cursor turret image

  Args:
    cfg (EasyDict): dictionary of configurations
  Returns:
    screen (pg.Surface): game screen
    clock (pg.time.Clock): game clock
    cursor_turret (pg.Surface): image of a turret for cursor
  """

  #initialise pygame
  pg.init()
  #create clock
  clock = pg.time.Clock()
  #create game window
  screen = pg.display.set_mode(
    (cfg.game.screen.width + cfg.game.screen.side_panel, cfg.game.screen.height))
  pg.display.set_caption(cfg.game.name)

  # initializing text fonts
  text_font = pg.font.SysFont("Consolas", 24, bold = True)
  large_font = pg.font.SysFont("Consolas", 36)

  # loading gui iamges
  heart_image = pg.image.load(cfg.game.screen.images.heart).convert_alpha()
  coin_image = pg.image.load(cfg.game.screen.images.coin).convert_alpha()
  logo_image = pg.image.load(cfg.game.screen.images.logo).convert_alpha()

  # loading cursor turret image
  cursor_turret = pg.image.load(cfg.game.turret.cursor_turret).convert_alpha()
  return screen, clock, cursor_turret


def draw_text(
  screen:pg.Surface, 
  text:str, 
  text_col:int, 
  x:int, 
  y:int, 
  large:bool=True
  ) -> None:
  """ 
  Function for outputting text onto the screen

  Args:
    screen (pg.Surface): game screen
    text (str): text to write on the screen
    text_col (int): index of column where to place text on the game table
    x (int): x coordinate of the text position
    y (int): y coordinate of the text position
    large (bool): whether to write large / normal text size
  """
  if large:
    img = large_font.render(text, True, text_col)
  else:
    img = text_font.render(text, True, text_col)
  screen.blit(img, (x, y))


def display_data(
  cfg:EasyDict, 
  screen:pg.Surface, 
  world:World
  ) -> Tuple[pg.Surface, World]:
  """ 
  Displaying data on side panel for game statistics

  Args:
    cfg (EasyDict): dictionary of configurations
    screen (pg.Surface): game screen
    world (World): class of the game world 
  Returns:
    screen (pg.Surface): game screen
    world (World): class of the game world 
  """
  # draw panel
  pg.draw.rect(screen, "maroon", 
    (cfg.game.screen.width, 0, cfg.game.screen.side_panel, cfg.game.screen.height))
  pg.draw.rect(screen, "grey0", 
    (cfg.game.screen.width, 0,cfg.game.screen.side_panel, 400), 2)
  screen.blit(logo_image, (cfg.game.screen.width, 400))
  # display data
  draw_text(screen, "LEVEL: " + str(world.level), 
    "grey100", cfg.game.screen.width + 10, 10, large=False)
  screen.blit(heart_image, (cfg.game.screen.width + 10, 35))
  draw_text(screen, str(world.health), 
    "grey100", cfg.game.screen.width + 50, 40, large=False)
  screen.blit(coin_image, (cfg.game.screen.width + 10, 65))
  draw_text(screen, str(world.money), 
    "grey100", cfg.game.screen.width + 50, 70, large=False)

  return screen, world


def game_result_plot(screen:pg.Surface, game_outcome:bool) -> None:
  """ 
  Game results plot

  Args:
    screen (pg.Surface): game screen
    game_outcome (bool): whether the result of the game is win or loss
  """
  pg.draw.rect(screen, "dodgerblue", (200, 200, 400, 200), border_radius = 30)
  if game_outcome == -1:
    draw_text(screen, "GAME OVER", "grey0", 310, 230, large=True)
  elif game_outcome == 1:
    draw_text(screen, "YOU WIN!", "grey0", 315, 230, large=True)