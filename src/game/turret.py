import pygame as pg
import math


class Turret(pg.sprite.Sprite):
  def __init__(self, cfg, tile_x, tile_y):
    pg.sprite.Sprite.__init__(self)
    self.cfg = cfg
    self.upgrade_level = 1
    self.range = [rang for rang, coold in self.cfg.game.turret.upgrades]
    self.cooldown = [coold for rang, coold in self.cfg.game.turret.upgrades]
    self.last_shot = pg.time.get_ticks()
    self.selected = False
    self.target = None

    #position variables
    self.tile_x = tile_x
    self.tile_y = tile_y
    #calculate center coordinates
    self.x = (self.tile_x + 0.5) * self.cfg.game.screen.tile_size
    self.y = (self.tile_y + 0.5) * self.cfg.game.screen.tile_size

    #animation variables
    self.sprite_sheets = [
      pg.image.load(path).convert_alpha() for path in cfg.turrent_path]
    self.animation_list = self.load_images(self.sprite_sheets[self.upgrade_level - 1])
    self.frame_index = 0
    self.update_time = pg.time.get_ticks()

    #update image
    self.angle = 90
    self.original_image = self.animation_list[self.frame_index]
    self.image = pg.transform.rotate(self.original_image, self.angle)
    self.rect = self.image.get_rect()
    self.rect.center = (self.x, self.y)

    #create transparent circle showing range
    self.range_image = pg.Surface((self.range * 2, self.range * 2))
    self.range_image.fill((0, 0, 0))
    self.range_image.set_colorkey((0, 0, 0))
    pg.draw.circle(self.range_image, "grey100", (self.range, self.range), self.range)
    self.range_image.set_alpha(100)
    self.range_rect = self.range_image.get_rect()
    self.range_rect.center = self.rect.center

  def load_images(self, sprite_sheet):
    #extract images from spritesheet
    size = sprite_sheet.get_height()
    animation_list = []
    for x in range(self.cfg.game.turret.animation_steps):
      temp_img = sprite_sheet.subsurface(x * size, 0, size, size)
      animation_list.append(temp_img)
    return animation_list

  def update(self, enemy_group, world):
    #if target picked, play firing animation
    if self.target:
      self.play_animation()
    else:
      #search for new target once turret has cooled down
      if pg.time.get_ticks() - self.last_shot > (self.cooldown / world.game_speed):
        self.pick_target(enemy_group)

  def pick_target(self, enemy_group):
    #find an enemy to target
    x_dist = 0
    y_dist = 0
    #check distance to each enemy to see if it is in range
    for enemy in enemy_group:
      if enemy.health > 0:
        x_dist = enemy.pos[0] - self.x
        y_dist = enemy.pos[1] - self.y
        dist = math.sqrt(x_dist ** 2 + y_dist ** 2)
        if dist < self.range:
          self.target = enemy
          self.angle = math.degrees(math.atan2(-y_dist, x_dist))
          #damage enemy
          self.target.health -= self.cfg.game.turret.damage
          break

  def play_animation(self):
    #update image
    self.original_image = self.animation_list[self.frame_index]
    #check if enough time has passed since the last update
    if pg.time.get_ticks() - self.update_time > self.cfg.game.turret.animation_delay:
      self.update_time = pg.time.get_ticks()
      self.frame_index += 1
      #check if the animation has finished and reset to idle
      if self.frame_index >= len(self.animation_list):
        self.frame_index = 0
        #record completed time and clear target so cooldown can begin
        self.last_shot = pg.time.get_ticks()
        self.target = None

  def upgrade(self):
    self.upgrade_level += 1
    self.range = [rang for rang, coold in self.cfg.game.turret.upgrades]
    self.cooldown = [coold for rang, coold in self.cfg.game.turret.upgrades]
    #upgrade turret image
    self.animation_list = self.load_images(self.sprite_sheets[self.upgrade_level - 1])
    self.original_image = self.animation_list[self.frame_index]

    #upgrade range circle
    self.range_image = pg.Surface((self.range * 2, self.range * 2))
    self.range_image.fill((0, 0, 0))
    self.range_image.set_colorkey((0, 0, 0))
    pg.draw.circle(self.range_image, "grey100", (self.range, self.range), self.range)
    self.range_image.set_alpha(100)
    self.range_rect = self.range_image.get_rect()
    self.range_rect.center = self.rect.center

  def draw(self, surface):
    self.image = pg.transform.rotate(self.original_image, self.angle - 90)
    self.rect = self.image.get_rect()
    self.rect.center = (self.x, self.y)
    surface.blit(self.image, self.rect)
    if self.selected:
      surface.blit(self.range_image, self.range_rect)


def create_turret(cfg, mouse_pos, world, turret_group):
  mouse_tile_x = mouse_pos[0] // cfg.game.screen.tile_size
  mouse_tile_y = mouse_pos[1] // cfg.game.screen.tile_size
  #calculate the sequential number of the tile
  mouse_tile_num = (mouse_tile_y * cfg.game.screen.cols) + mouse_tile_x
  #check if that tile is grass
  if world.tile_map[mouse_tile_num] == 7:
    #check that there isn't already a turret there
    space_is_free = True
    for turret in turret_group:
      if (mouse_tile_x, mouse_tile_y) == (turret.tile_x, turret.tile_y):
        space_is_free = False
    #if it is a free space then create turret
    if space_is_free is True:
      new_turret = Turret(cfg, mouse_tile_x, mouse_tile_y)
      turret_group.add(new_turret)
      #deduct cost of turret
      world.money -= cfg.game.turret.buy_cost


def select_turret(cfg, mouse_pos, turret_group):
  mouse_tile_x = mouse_pos[0] // cfg.game.screen.tile_size
  mouse_tile_y = mouse_pos[1] // cfg.game.screen.tile_size
  for turret in turret_group:
    if (mouse_tile_x, mouse_tile_y) == (turret.tile_x, turret.tile_y):
      return turret

def clear_selection(turret_group):
  for turret in turret_group:
    turret.selected = False