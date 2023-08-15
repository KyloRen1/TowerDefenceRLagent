import pygame as pg

class World():
    def __init__(self, data, map_image):
        self.level_data = data
        self.image = map_image
        self.waypoints = list()
        self.tile_map = list()
        self.process_data()

    def process_data(self):
        for layer in self.level_data['layers']:
            if layer['name'] == 'tilemap':
                self.tile_map = layer['data']
            elif layer['name'] == 'waypoints':
                for obj in layer['objects']:
                    waypoint_data = obj['polyline']
                    self.process_waypoints(waypoint_data)

    def process_waypoints(self, data):
        for point in data:
            temp_x = point.get('x')
            temp_y = point.get('y')
            self.waypoints.append((temp_x, temp_y))

    def draw(self, surface):
        surface.blit(self.image, (0, 0))

