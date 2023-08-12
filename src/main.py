import click 
import pygame as pg
from .enemy import Enemy
from .world import World

from src.utils import load_config, create_game_screen




@click.command(help="")
@click.option("--cfg", type=str, help="config file path")
def main(cfg):
    cfg = load_config(cfg)

    screen, clock = create_game_screen(cfg)

    enemy_group = pg.sprite.Group()

    waypoints = [
        (100, 100),
        (400, 200),
        (400, 100),
        (200, 300)
    ]

    map_image = pg.image.load('').convert_alpha()
    world = World(map_image)

    enemy_image = pg.image.load('src/assets/images/enemies/enemy_1.png').convert_alpha()
    enemy = Enemy(waypoints, enemy_image)
    enemy_group.add(enemy)
    print(enemy)

    run = True
    while run:
        # event handler

        world.draw(screen)
        
        clock.tick(cfg.game.fps)
        screen.fill(cfg.game.screen.color)

        pg.draw.lines(screen, 'grey0', False, waypoints)

        enemy_group.update()
        enemy_group.draw(screen)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                run = False

        pg.display.flip()

    pg.quit()


if __name__ == '__main__':
    main()