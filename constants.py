import numpy as np

# Cityscapes 
LABEL_COLORS = np.array([
    (255, 255, 255), # 0: NONE
    (128, 64, 128),  # 1: Roads
    (244, 35, 232),  # 2: Sidewalks
    (70, 70, 70),    # 3: Buildings
    (102, 102, 156), # 4: Walls
    (190, 153, 153), # 5: Fences
    (153, 153, 153), # 6: Poles
    (250, 170, 30),  # 7: TrafficLight
    (220, 220, 0),   # 8: TrafficSigns
    (107, 142, 35),  # 9: Vegetation
    (152, 251, 152), # 10: Terrain
    (70, 130, 180),  # 11: Sky
    (220, 20, 60),   # 12: Pedestrians
    (255, 0, 0),     # 13: Rider
    (0, 0, 142),     # 14: Car
    (0, 0, 70),      # 15: Truck
    (0, 60, 100),    # 16: Bus
    (0, 80, 100),    # 17: Train
    (0, 0, 230),     # 18: Motorcycle
    (119, 11, 32),   # 19: Bicycle
    (110, 190, 160), # 20: Static
    (170, 120, 50),  # 21: Dynamic
    (55, 90, 80),    # 22: Other
    (45, 60, 150),   # 23: Water
    (157, 234, 50),  # 24: RoadLines
    (81, 0, 81),     # 25: Ground
    (150, 100, 100), # 26: Bridge
    (230, 150, 140), # 27: RailTrack
    (180, 165, 180), # 28: GuardRail
]) / 255.0  # normalize to [0, 1]

#BICYCLE_TYPES = ['CrossBike', 'Harley', 'kawasaki', 'Leisure', 'Bike', 'Vespa', 'Yamaha']
