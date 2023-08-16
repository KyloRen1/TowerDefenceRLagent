import pygame as pg 

class Turret(pg.sprite.Sprite):
    def __init__(self, cfg, sprite_sheet, tile_x, tile_y):
        super().__init__()
        self.cfg = cfg

        self.selected = False

        self.range = 90
        self.cooldown = 0
        self.last_shot = pg.time.get_ticks()

        self.sprite_sheet = sprite_sheet
        self.animation_list = self.load_images()
        self.frame_idx = 0
        self.update_time = pg.time.get_ticks()

        self.image = self.animation_list[self.frame_idx] 
        self.rect = self.image.get_rect()

        self.tile_x = tile_x 
        self.tile_y = tile_y

        self.x = (tile_x + 0.5) * cfg.game.screen.tile_size
        self.y = (tile_y + 0.5) * cfg.game.screen.tile_size
        self.rect.center = (self.x, self.y)

        self.range_image = pg.Surface((self.range * 2, self.range * 2))
        self.range_image.fill((0, 0, 0))
        self.range_image.set_colorkey((0, 0, 0))
        pg.draw.circle(self.range_image, "grey100", (self.range, self.range), self.range)
        self.range_image.set_alpha(100)
        self.range_rect = self.range_image.get_rect()
        self.range_rect.center = self.rect.center

    def load_images(self):
        size = self.sprite_sheet.get_height()
        animation_list = list()
        for x in range(self.cfg.game.turret.animation_steps):
            temp_img = self.sprite_sheet.subsurface(x * size, 0, size, size)
            animation_list.append(temp_img)
        return animation_list

    def update(self):
        # search for new target after cooling down
        if pg.time.get_ticks() - self.last_shot > self.cfg.game.turret.animation_delay:
            self.play_animation()


    def play_animation(self):
        # update image
        self.image = self.animation_list[self.frame_idx]

        if pg.time.get_ticks() - self.update_time < self.cfg.game.turret.animation_delay:
            self.update_time = pg.time.get_ticks()
            self.frame_idx += 1 
            if self.frame_idx >= len(self.animation_list):
                self.frame_idx = 0
                self.last_shot = pg.time.get_ticks()


    def draw(self, surface):
        surface.blit(self.image, self.rect)
        if self.selected:
            surface.blit(self.range_image, self.range_rect)


def create_turret(cfg, world, mouse_pos, turret_sheet, turret_group):
    mouse_tile_x = mouse_pos[0] // cfg.game.screen.tile_size
    mouse_tile_y = mouse_pos[1] // cfg.game.screen.tile_size
    mouse_tile_num = (mouse_tile_y * cfg.game.screen.cols) + mouse_tile_x
    if world.tile_map[mouse_tile_num] == 7:
        space_is_free = True
        for turret in turret_group:
            if (mouse_tile_x, mouse_tile_y) == (turret.tile_x, turret.tile_y):
                space_is_free = False
        
        if space_is_free:
            turret = Turret(cfg, turret_sheet, mouse_tile_x, mouse_tile_y)
            turret_group.add(turret)
    return turret_group


def select_turret(cfg, mouse_pos, turret_group):
    mouse_tile_x = mouse_pos[0] // cfg.game.screen.tile_size
    mouse_tile_y = mouse_pos[1] // cfg.game.screen.tile_size

    for turret in turret_group:
        if (mouse_tile_x, mouse_tile_y) == (turret.tile_x, turret.tile_y):
            return turret

def clear_selection(turret_group):
    for turret in turret_group:
        turret.selected = False