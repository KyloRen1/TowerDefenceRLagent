import pygame as pg

#load fonts for displaying text on the screen
text_font = None
large_font = None

# coin 
coin_image = None

def create_game_window(cfg):
  global text_font, large_font
  #initialise pygame
  pg.init()
  #create clock
  clock = pg.time.Clock()
  #create game window
  screen = pg.display.set_mode(
    (cfg.game.screen.width + cfg.game.screen.side_panel, cfg.game.screen.height))
  pg.display.set_caption(cfg.game.name)

  text_font = pg.font.SysFont("Consolas", 24, bold = True)
  large_font = pg.font.SysFont("Consolas", 36)
  return screen, clock

#function for outputting text onto the screen
def draw_text(screen, text, text_col, x, y, large_font=True):
  if large_font:
    img = large_font.render(text, True, text_col)
  else:
    img = text_font.render(text, True, text_col)
  screen.blit(img, (x, y))


def display_data(cfg, screen, world):
  global coin_image
  # loading gui iamges
  heart_image = pg.image.load(cfg.game.screen.images.heart).convert_alpha()
  coin_image = pg.image.load(cfg.game.screen.images.coin).convert_alpha()
  logo_image = pg.image.load(cfg.game.screen.images.logo).convert_alpha()
  # draw panel
  pg.draw.rect(screen, "maroon", 
    (cfg.game.screen.width, 0, cfg.game.screen.side_panel, cfg.game.screen.height))
  pg.draw.rect(screen, "grey0", 
    (cfg.game.screen.width, 0,cfg.game.screen.side_panel, 400), 2)
  screen.blit(logo_image, (cfg.game.screen.width, 400))
  # display data
  draw_text(screen, "LEVEL: " + str(world.level), 
    "grey100", cfg.game.screen.width + 10, 10, large_font=False)
  screen.blit(heart_image, (cfg.game.screen.width + 10, 35))
  draw_text(screen, str(world.health), 
    "grey100", cfg.game.screen.width + 50, 40, large_font=False)
  screen.blit(coin_image, (cfg.game.screen.width + 10, 65))
  draw_text(screen, str(world.money), 
    "grey100", cfg.game.screen.width + 50, 70, large_font=False)

  return screen, world