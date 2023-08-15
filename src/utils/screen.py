import pygame as pg


def create_game_screen(config):
    pg.init()
    pg.display.set_caption(config.game.name)

    screen = pg.display.set_mode((
        config.game.screen.width + config.game.screen.side_panel, 
        config.game.screen.height
    ))

    clock = pg.time.Clock()
    return screen, clock