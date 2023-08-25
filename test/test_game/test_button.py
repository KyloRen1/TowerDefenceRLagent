import unittest
import pygame as pg 
from easydict import EasyDict

from src.game.button import Button

class ButtonTest(unittest.TestCase):
    def setUp(self):
        print('\n Setting up the test')
        self.cfg = EasyDict({
            'game': {
                'screen': {
                    'width': 1020,
                    'height': 720
                },
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
                    },
                }

            }
        })
        pg.init()
        self.screen = pg.display.set_mode((
            self.cfg.game.screen.width, 
            self.cfg.game.screen.height))

    def test_buy_turret_button(self):
        button =  Button(self.cfg, 'buy_turret', single_click=True)
        self.assertEqual(button.draw(self.surface), False)


if __name__ == '__main__':
    unittest.main()