import click
import pygame as pg

from src.game.utils import (load_config, 
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
  # loading config
  cfg = load_config(cfg)# initialise pygame
  # initializing game
  screen, clock, cursor_turret = create_game_window(cfg)

  # game variabless
  game_over:        bool = False
  game_outcome:     int  = 0 # -1 is loss & 1 is win
  level_started:    bool = False
  last_enemy_spawn: int  = pg.time.get_ticks()
  placing_turrets:  bool = False
  selected_turret:  bool = None

  # create world
  world = World(cfg)
  world.process_data()
  world.process_enemies()

  # create buttons
  turret_button       = Button(cfg, 'buy_turret', single_click=True)
  cancel_button       = Button(cfg, 'cancel', single_click=True)
  upgrade_button      = Button(cfg, 'upgrade', single_click=True)  
  begin_button        = Button(cfg, 'begin', single_click=True)
  restart_button      = Button(cfg, 'restart', single_click=True)
  fast_forward_button = Button(cfg, 'fast_forward', single_click=False)

  # create groups
  enemy_group = pg.sprite.Group()
  turret_group = pg.sprite.Group()

  # game loop
  run: bool = True
  while run:
    clock.tick(cfg.game.fps)

    #########################
    # UPDATING SECTION
    #########################

    if game_over is False:
      # check if player has lost
      if world.health <= 0:
        game_over = True
        game_outcome = -1 # loss
      # check if player has won
      if world.level > cfg.game.levels:
        game_over = True
        game_outcome = 1 # win

      # update groups
      enemy_group.update(world)
      turret_group.update(enemy_group, world)

      # highlight selected turret
      if selected_turret:
        selected_turret.selected = True

    #########################
    # DRAWING SECTION
    #########################
    # draw level
    world.draw(screen)

    # draw groups
    enemy_group.draw(screen)
    for turret in turret_group:
      turret.draw(screen)

    screen, world = display_data(cfg, screen, world)

    #########################
    # GAME PROGRESS SECTION
    #########################

    if game_over is False:
      # check if the level has been started or not
      if level_started is False:
        if begin_button.draw(screen):
          level_started = True
      else:
        # fast forward option
        world.game_speed = 1
        if fast_forward_button.draw(screen):
          world.game_speed = 2
        # spawn enemies
        if pg.time.get_ticks() - last_enemy_spawn > cfg.game.enemy.spawn_cooldown:
          if world.spawned_enemies < len(world.enemy_list):
            enemy_type = world.enemy_list[world.spawned_enemies]
            enemy = Enemy(cfg, enemy_type, world.waypoints)
            enemy_group.add(enemy)
            world.spawned_enemies += 1
            last_enemy_spawn = pg.time.get_ticks()

      # check if the wave is finished
      if world.check_level_complete() is True:
        world.money += cfg.game.level_complete_reward
        world.level += 1
        level_started = False
        last_enemy_spawn = pg.time.get_ticks()
        world.reset_level()
        world.process_enemies()

      # draw buttons
      # button for placing turrets
      # for the "turret button" show cost of turret and draw the button
      draw_text(screen, str(cfg.game.turret.buy_cost), 
        "grey100", cfg.game.screen.width + 215, 135, large=False)
      if turret_button.draw(screen):
        placing_turrets = True
      # if placing turrets then show the cancel button as well
      if placing_turrets is True:
        # show cursor turret
        cursor_rect = cursor_turret.get_rect()
        cursor_pos = pg.mouse.get_pos()
        cursor_rect.center = cursor_pos
        if cursor_pos[0] <= cfg.game.screen.width:
          screen.blit(cursor_turret, cursor_rect)
        if cancel_button.draw(screen):
          placing_turrets = False
      # if a turret is selected then show the upgrade button
      if selected_turret:
        # if a turret can be upgraded then show the upgrade button
        if selected_turret.upgrade_level < cfg.game.turret.levels:
          # show cost of upgrade and draw the button
          draw_text(screen, str(cfg.game.turret.upgrade_cost), 
            "grey100", cfg.game.screen.width + 215, 195, large=False)
          if upgrade_button.draw(screen):
            if world.money >= cfg.game.turret.upgrade_cost:
              selected_turret.upgrade()
              world.money -= cfg.game.turret.upgrade_cost
    else:
      game_result_plot(screen, game_outcome)
      if restart_button.draw(screen):
        game_over = False
        level_started = False
        placing_turrets = False
        selected_turret = None
        last_enemy_spawn = pg.time.get_ticks()
        world = World(cfg)
        world.process_data()
        world.process_enemies()
        # empty groups
        enemy_group.empty()
        turret_group.empty()

    #########################
    # EVENT HANDLING SECTION
    #########################
    for event in pg.event.get():
      # quit program
      if event.type == pg.QUIT:
        run = False
      # mouse click
      if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
        mouse_pos = pg.mouse.get_pos()
        # check if mouse is on the game area
        if mouse_pos[0] < cfg.game.screen.width and mouse_pos[1] < cfg.game.screen.height:
          # clear selected turrets
          selected_turret = None
          clear_selection(turret_group)
          if placing_turrets is True:
            # check if there is enough money for a turret
            if world.money >= cfg.game.turret.buy_cost:
              create_turret(cfg, mouse_pos, world, turret_group)
          else:
            selected_turret = select_turret(cfg, mouse_pos, turret_group)

    # update display
    pg.display.flip()

  pg.quit()


if __name__ == '__main__':
  main()