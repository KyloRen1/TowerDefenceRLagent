import click 
import json
import pygame as pg
from .enemy import Enemy
from .world import World
from .turret import Turret, create_turret

from src.utils import load_config, create_game_screen




@click.command(help="")
@click.option("--cfg", type=str, help="config file path")
def main(cfg):
    cfg = load_config(cfg)

    screen, clock = create_game_screen(cfg)

    enemy_group = pg.sprite.Group()
    turret_group = pg.sprite.Group()

    with open('src/assets/levels/level.tmj') as file:
        world_data = json.load(file)

    map_image = pg.image.load('src/assets/images/level.png').convert_alpha()
    world = World(world_data, map_image)

    cursor_turret = pg.image.load('src/assets/images/turrets/cursor_turret.png').convert_alpha()

    enemy_image = pg.image.load('src/assets/images/enemies/enemy_1.png').convert_alpha()
    enemy = Enemy(world.waypoints, enemy_image)
    enemy_group.add(enemy)

    run = True
    while run:
        # event handler
        clock.tick(cfg.game.fps)
        screen.fill(cfg.game.screen.color)
        world.draw(screen)


        enemy_group.update()
        enemy_group.draw(screen)
        turret_group.draw(screen)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False
            if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pg.mouse.get_pos()
                if mouse_pos[0] < cfg.game.screen.width and mouse_pos[1] < cfg.game.screen.height:
                    turret_group = create_turret(cfg, world, mouse_pos, cursor_turret, turret_group)

        pg.display.flip()

    pg.quit()


if __name__ == '__main__':
    main()