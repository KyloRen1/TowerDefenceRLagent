import pytest
import pygame as pg 
from easydict import EasyDict

from src.game.button import Button

@pytest.fixture 
def config():
    cfg = EasyDict({
        'game': {
            'screen': {'width': 1020, 'height': 720},
            'button': {
                'buy_turret': {
                    'image': 'src/assets/images/buttons/buy_turret.png',
                    'coord': (750, 120)
                },
                'cancel': {
                    'image': 'src/assets/images/buttons/cancel.png',
                    'coord': (770, 180)
                },
                'upgrade': {
                    'image': 'src/assets/images/buttons/upgrade_turret.png',
                    'coord': (725, 180)
                },
                'begin': {
                    'image': 'src/assets/images/buttons/begin.png',
                    'coord': (780, 300)
                },
                'restart': {
                    'image': 'src/assets/images/buttons/restart.png',
                    'coord': (310, 300)
                },
                'fast_forward': {
                    'image': 'src/assets/images/buttons/fast_forward.png',
                    'coord': (770, 300)
                }}}})
    return cfg

@pytest.fixture
def game_screen(config):
    pg.init()
    screen = pg.display.set_mode((config.game.screen.width, config.game.screen.height))
    return screen

def test_buy_turret_button(config, game_screen):
    button =  Button(config, 'buy_turret', single_click=True)
    # TODO write test

def test_cancel_button(config, game_screen):
    button =  Button(config, 'cancel', single_click=True)
    # TODO write test

def test_upgrade_button(config, game_screen):
    button =  Button(config, 'upgrade', single_click=True)
    # TODO write test

def test_begin_button(config, game_screen):
    button =  Button(config, 'begin', single_click=True)
    # TODO write test

def test_restart_button(config, game_screen):
    button =  Button(config, 'restart', single_click=True)
    # TODO write test

def test_fast_forward_button(config, game_screen):
    button =  Button(config, 'fast_forward', single_click=False)
    # TODO write test

