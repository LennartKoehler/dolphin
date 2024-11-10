# (0,0,0) top left corner of your Z-stack:
# x goes from left to right, y from top to bottom and z from front to back 
# in your Z-stack projected on this illustration

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse

def plot_cube_with_subcubes(cube_size=10, subcube_size=3):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])  # Equal scaling in all directions

    # Set the boundaries for the main cube
    ax.set_xlim([0, cube_size])
    ax.set_ylim([0, cube_size])
    ax.set_zlim([0, cube_size])

    # Counter for the subcubes
    subcube_count = 0
    
    # Loops for the subcubes, allowing the last ones to protrude beyond the edge
    for y in range(0, cube_size, subcube_size):
        for x in range(0, cube_size, subcube_size):
            for z in range(0, cube_size, subcube_size):
                # Calculate the corners of the subcube
                subcube_corners = [
                    [x, y, z],
                    [min(x + subcube_size, cube_size), y, z],
                    [min(x + subcube_size, cube_size), min(y + subcube_size, cube_size), z],
                    [x, min(y + subcube_size, cube_size), z],
                    [x, y, min(z + subcube_size, cube_size)],
                    [min(x + subcube_size, cube_size), y, min(z + subcube_size, cube_size)],
                    [min(x + subcube_size, cube_size), min(y + subcube_size, cube_size), min(z + subcube_size, cube_size)],
                    [x, min(y + subcube_size, cube_size), min(z + subcube_size, cube_size)]
                ]
                
                # Draw the subcube
                faces = [
                    [subcube_corners[0], subcube_corners[1], subcube_corners[2], subcube_corners[3]],
                    [subcube_corners[4], subcube_corners[5], subcube_corners[6], subcube_corners[7]],
                    [subcube_corners[0], subcube_corners[1], subcube_corners[5], subcube_corners[4]],
                    [subcube_corners[2], subcube_corners[3], subcube_corners[7], subcube_corners[6]],
                    [subcube_corners[1], subcube_corners[2], subcube_corners[6], subcube_corners[5]],
                    [subcube_corners[4], subcube_corners[7], subcube_corners[3], subcube_corners[0]]
                ]
                
                ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.01))
                
                # Add numbering at the center of the subcube
                center = [x + subcube_size / 2, y + subcube_size / 2, z + subcube_size / 2]
                ax.text(center[0], center[1], center[2], str(subcube_count), color="black", ha='center')
                
                subcube_count += 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # ax.view_init(elev=110, azim=0)
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a cube with subcubes inside it.")
    parser.add_argument('--cube_size', type=int, default=10, help='Size of the main cube')
    parser.add_argument('--subcube_size', type=int, default=3, help='Size of each subcube')
    
    args = parser.parse_args()
    
    plot_cube_with_subcubes(cube_size=args.cube_size, subcube_size=args.subcube_size)

