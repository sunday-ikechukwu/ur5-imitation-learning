import random
import numpy as np

def randomize_cube_position(sim, cube_handle):
    '''Randomize the position of the cube within the reachable workspace of the robot.'''
    # Implementation for randomizing cube position
    x = random.uniform(0.46, 1.10)
    y = random.uniform(-1.11, -0.76)
    z = 0.38  # Keep the cube on the table
    sim.setObjectPosition(cube_handle, -1, [x, y, z])

    return np.array([x, y, z])