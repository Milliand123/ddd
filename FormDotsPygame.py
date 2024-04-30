import random

import pygame
import numpy as np
from numba import jit
from neural_network import NeuralNetwork


class Point:
    def __init__(self, x: int, y: int, type: int):
        self.x = x
        self.y = y
        self.type = type


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


@jit(forceobj=True)
def main():
    pygame.init()

    w, h = 1280, 720
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    nn = NeuralNetwork(0.01, sigmoid, dsigmoid, [2, 5, 5, 2])
    points = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouseX, mouseY = pygame.mouse.get_pos()
                if event.button == 2:
                    closest_point = None
                    closest_distance = float('inf')
                    for point in points:
                        distance = np.sqrt((point.x - mouseX) ** 2 + (point.y - mouseY) ** 2)
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_point = point
                    if closest_point is not None and closest_distance < 20:
                        points.remove(closest_point)
                else:
                    ptype = 0 if event.button == 1 else 1
                    points.append(Point(mouseX, mouseY, ptype))

        screen.fill((0, 0, 0))
        if points:
            for _ in range(10000):
                p = random.choice(points)
                nx = p.x / w - 0.5
                ny = p.y / h - 0.5
                nn.feed_forward(np.array([nx, ny]))
                targets = [0, 1] if p.type == 1 else [1, 0]
                nn.backpropagation(targets)

        for i in range(w // 8):
            for j in range(h // 8):
                nx = i / w * 8 - 0.5
                ny = j / h * 8 - 0.5
                outputs = nn.feed_forward(np.array([nx, ny]))
                green = np.clip(outputs[0] - outputs[1] + 0.5, 0, 1)
                blue = 1 - green
                green = 0.3 + green * 0.5
                blue = 0.5 + blue * 0.5
                color = (int(100), int(green * 255), int(blue * 255))
                pygame.draw.rect(screen, color, (i * 8, j * 8, 8, 8))

        for point in points:
            pygame.draw.circle(screen, (255, 255, 255), (point.x, point.y), 20, 0)
            color = (0, 255, 0) if point.type == 0 else (0, 0, 255)
            pygame.draw.circle(screen, color, (point.x, point.y), 16, 0)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


main()
