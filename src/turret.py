import pygame as pg 

class Turret(pg.sprite.Sprite):
    def __init__(self, cfg, image, tile_x, tile_y):
        super().__init__()
        self.image = image 
        self.rect = self.image.get_rect()

        self.tile_x = tile_x 
        self.tile_y = tile_y

        self.x = (tile_x + 0.5) * cfg.game.screen.tile_size
        self.y = (tile_y + 0.5) * cfg.game.screen.tile_size
        self.rect.center = (self.x, self.y)


def create_turret(cfg, world, mouse_pos, cursor_turret, turret_group):
    mouse_tile_x = mouse_pos[0] // cfg.game.screen.tile_size
    mouse_tile_y = mouse_pos[1] // cfg.game.screen.tile_size
    mouse_tile_num = (mouse_tile_y * cfg.game.screen.cols) + mouse_tile_x
    if world.tile_map[mouse_tile_num] == 7:
        space_is_free = True
        for turret in turret_group:
            if (mouse_tile_x, mouse_tile_y) == (turret.tile_x, turret.tile_y):
                space_is_free = False
        
        if space_is_free:
            turret = Turret(cfg, cursor_turret, mouse_tile_x, mouse_tile_y)
            turret_group.add(turret)
    return turret_group