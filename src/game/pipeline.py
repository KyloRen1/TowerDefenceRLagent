
import click
import pygame as pg 

from src.game.turret import Turret
from src.game.world import World
from src.game.enemy import Enemy
from src.game.utils import (
    load_config, 
    create_game_window, 
    display_data
)

from easydict import EasyDict
from typing import Tuple


class TowerDefence:
    def __init__(self, cfg:EasyDict) -> None:
        ''' initializing game and game variables '''
        self.cfg = cfg
        self.frame = 0

        # init of screen and clock
        self.screen, self.clock, _ = create_game_window(cfg)

        # init of groups
        self.enemy_group = pg.sprite.Group()
        self.turret_group = pg.sprite.Group()

        # reset world and variables
        self.reset()


    def reset(self) -> None:
        # reset game variables
        self.game_over = False 
        self.game_outcome = 0
        self.level_started = False 
        self.placing_turrets = False 
        self.selected_turret = None
        self.last_enemy_spawn = pg.time.get_ticks()
        self.last_level = pg.time.get_ticks()

        # reset world
        self.world = World(self.cfg)

        # reset groups
        self.enemy_group.empty()
        self.turret_group.empty()


    def _create_turret(self, tile_x, tile_y):
        # check if that tile is grass
        if (self.world.tile_map[tile_x + tile_y] == 7 and 
            self.world.money >= self.cfg.game.turret.buy_cost):
            # check that there isn't already a turret there
            space_is_free = True 
            for turret in self.turret_group:
                if (tile_x, tile_y) == (turret.tile_x, turret.tile_y):
                    space_is_free = False
            # if it is a free space then create turret
            if space_is_free:
                new_turret = Turret(self.cfg, tile_x, tile_y)
                self.turret_group.add(new_turret)
                # deduct cost of turret
                self.world.money -= self.cfg.game.turret.buy_cost


    def _upgrade_turret(self, tile_x, tile_y):
        selected_turret = None
        for turret in self.turret_group:
            if (tile_x, tile_y) == (turret.tile_x, turret.tile_y):
                selected_turret = turret
        # check if that is possible to upgrade selected turret
        if selected_turret.upgrade_level < self.cfg.game.turret.levels:
            # check if there is enough money to upgrade
            if self.world.money >= self.cfg.game.turret.upgrade_cost:
                selected_turret.upgrade()
                # deduct cost of turret upgrade
                self.world.money -= self.cfg.game.turret.upgrade_cost


    def _check_game_over(self):
        if self.game_over is False:
            # check if player has lost
            if self.world.health <= 0:
                self.game_over = True 
                self.game_outcome = -1 # loss
            if self.world.level > self.cfg.game.levels:
                self.game_over = True 
                self.game_outcome = 1 # win


    def _check_level_finished(self):
        if self.world.check_level_complete():
            self.world.money += self.cfg.game.level_complete_reward
            self.world.level += 1
            self.level_started = False
            self.last_enemy_spawn = pg.time.get_ticks()
            self.last_level = pg.time.get_ticks()
            self.world.reset_level()
            self.world.process_enemies()


    def _game_process(self):
        # checking is game is over
        self._check_game_over()
        if self.level_started is False:
            # starting level
            if (pg.time.get_ticks() - self.last_level > 
                    self.cfg.game.level_spawn_cooldown):
                self.level_started = True
        else:
            # starting enemy spawn
            if (pg.time.get_ticks() - self.last_enemy_spawn > 
                    self.cfg.game.enemy.spawn_cooldown):
                if self.world.spawned_enemies < len(self.world.enemy_list):
                    enemy_type = self.world.enemy_list[self.world.spawned_enemies]
                    enemy = Enemy(self.cfg, enemy_type, self.world.waypoints)
                    self.enemy_group.add(enemy)
                    self.world.spawned_enemies += 1
                    self.last_enemy_spawn = pg.time.get_ticks()
            
            # checking if level finished
            self._check_level_finished()


    def step(self, action) -> Tuple[int, bool, int]:
        self.frame += 1
        
        # values of health and number of killed enemies 
        # at the beginning of the step
        step_start_health = self.world.health
        step_killed_enemies = self.world.killed_enemies

        # check if game quit 
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        # perform action TODO logic
        # if action is to buy turret
        tile_x, tile_y = 1, 1 
        self._create_turret(tile_x, tile_y)
        # if action upgrade
        #self._upgrade_turret(tile_x, tile_y)
        
        # process game step
        self._game_process()

        # update ui and clock
        self._update_ui()
        self.clock.tick(self.cfg.game.fps)
        
        # constructing reward
        if self.game_over:
            # reward if 100 and -100 for won and lost game, respectively
            reward = self.game_outcome * 100
        else:
            # if game is in progress, reward is the number of health lost 
            health_diff = self.world.health - step_start_health 
            # and the number of enemies killed in this step
            killed_enemies_diff = self.world.killed_enemies - step_killed_enemies 
            reward = health_diff + killed_enemies_diff
        
        return reward, self.game_over, self.world.health


    def _update_ui(self):
        # draw world on screen
        self.world.draw(self.screen)

        # update enemy and turret groups
        self.enemy_group.update(self.world)
        self.turret_group.update(self.enemy_group, self.world)
        
        # draw enemy and turret groups
        self.enemy_group.draw(self.screen)
        for turret in self.turret_group:
            turret.draw(self.screen)

        # update screen and world
        self.screen, self.world = display_data(
            self.cfg, self.screen, self.world)
        
        # update display
        pg.display.flip()    


@click.command(help="")
@click.option("--cfg", type=str, help="config file path")
def main(cfg):
    config = load_config(cfg)

    game = TowerDefence(config)

    counter = 0
    while True:
        action = None
        reward, done, score = game.step(action)
        print(counter, reward, done)

        if done:
            game.reset()

        counter += 1


if __name__ == '__main__':
    main()
