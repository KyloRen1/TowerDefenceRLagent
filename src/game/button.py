import pygame as pg
from easydict import EasyDict


class Button():
    def __init__(
            self,
            cfg: EasyDict,
            button_name: str,
            single_click: bool) -> None:
        self.image = pg.image.load(
            cfg.game.button[f"{button_name}"].image).convert_alpha()
        self.rect = self.image.get_rect()
        self.rect.topleft = cfg.game.button[f"{button_name}"].coord
        self.clicked = False
        self.single_click = single_click

    def draw(self, surface: pg.Surface) -> bool:
        action = False
        # get mouse position
        pos = pg.mouse.get_pos()
        # check mouseover and clicked conditions
        if self.rect.collidepoint(pos):
            if pg.mouse.get_pressed()[0] == 1 and self.clicked is False:
                action = True
                # if button is a single click type, then set clicked to True
                if self.single_click:
                    self.clicked = True

        if pg.mouse.get_pressed()[0] == 0:
            self.clicked = False
        # draw button on screen
        surface.blit(self.image, self.rect)
        return action
