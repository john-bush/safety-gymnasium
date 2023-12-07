import numpy as np
from noise import pnoise2
from matplotlib import pyplot as plt


def generate_perlin_noise(width, height, scale=0.07):
    noise_grid = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            noise_grid[i][j] = pnoise2(i / scale, j / scale)
    return noise_grid

def create_flow_field(noise_grid, angle_scale=1):
    flow_field = np.zeros(noise_grid.shape, dtype=float)
    for i in range(noise_grid.shape[0]):
        for j in range(noise_grid.shape[1]):
            angle = noise_grid[i][j] * np.pi * angle_scale
            flow_field[i][j] = angle
    return flow_field

def create_flow_field_from_noise(width, height, scale=10, angle_scale=1):
    noise_grid = generate_perlin_noise(width, height, scale)
    flow_field = create_flow_field(noise_grid, angle_scale)
    return flow_field

if __name__ == "__main__":
    flow_field = create_flow_field_from_noise(500, 500, 10, 2)
    # create a matplotlib figure and show flow field as a plot
    plt.figure() 
    plt.imshow(flow_field)
    plt.show()

