import itertools as it
import os
import numpy as np
import pandas as pd
from PIL import Image
from itertools import *

# Algorithm 1: The algorithm for preparing the dataset.
# 1 S ← predefined size intervals
sz_upper_bound = 160
sz_lower_bound = 5
sz_interval_length = sz_upper_bound - sz_lower_bound
sz_mid_point = int(sz_interval_length / 2)
bins = np.concatenate((pd.cut(np.arange(sz_lower_bound, sz_mid_point), 14, retbins=True)[1],
                       pd.cut(np.arange(sz_mid_point, sz_upper_bound), 5, retbins=True)[1]))
size_intervals = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]
# print(size_intervals, len(size_intervals))

# 2 D ← foregrounds of drone images
# drones_images = ["drone3", "drone2", "drone1"]
drones_images = os.listdir('./drones')

# 3 B ← foregrounds of bird images
bird_images = os.listdir('./birds')

# 4 V ← background videos
bg_video = os.listdir('./backgrounds')
bg_video_dim = (800,800)

# 5 R ← # of rows that the image will be divided into
num_rows = 12

# 6 C ← # of columns that the image will be divided into
num_cols = 10

# 7 G ← R × C grid
grid_img_col = int(bg_video_dim[0]/num_cols)
grid_img_row = int(bg_video_dim[1]/num_rows)

grid = [(row, col) for row in range(0, bg_video_dim[1], grid_img_row) for col in range(0, bg_video_dim[0],grid_img_col)]

plane_images = []  # Extra negative image examples

total_configurations = len(drones_images) * len(grid) * len(size_intervals) * len(bg_video)
max_configurations = 500

# Generator of possible configurations
configurations = it.product(drones_images, grid, size_intervals, bg_video)

gen = 0
saved_config_counter = 0
saved_configs = []

acceptance_ratio = float(max_configurations / total_configurations)

# for config in configurations:
for config in configurations:
    # 8 foreach (d, g, s, v) ∈ D × G × S × V do

    # print(config)

    # 9 ignore this configuration with probability
    # p = 1 − Max. allowed size/ Total size for all configurations , and continue
    accept_config = np.random.choice(a=[False, True],
                                     p=[1.0 - acceptance_ratio, acceptance_ratio])

    gen += 1  # Generation counter

    # if gen > 10000:
    #     break

    if not accept_config:
        # print(gen, accept_config)
        continue
    else:
        # TODO this shouldn't be random? why? does it matter?

        # Unpack configuration : (d, g, s, v)
        d, g, s, v = config

        s0 = np.random.uniform(low=s[0], high=s[1])
        # print(s, s0)
        b = np.random.choice(bird_images)
        #TODO RANDOM POINT x,y = g range(x, x+80), range(y,y+80)

        print(d, g, s, v)
        # 10 draw a random position p0 in g
        # 11 draw a random size s0 for smaller edge of the drone from s
        # 12 draw a random frame f0 from v
        # 13 resize d with respect to s0
        # 14 overlay f0 with d in position p0

        # 15 draw (p1, s1, f1) in the same way
        # 16 draw a random bird b0 from B
        # 17 draw (pb,0, sb,0) for bird where sb,0 is drawn from smaller half of S
        # 18 resize d with respect to s1
        # 19 overlay f1 with d in position p1
        # 20 resize b0 with respect to sb,0
        # 21 overlay f1 with b0 in position pb,0

        # 22 draw (p2, s2, f2) in the same way
        # 23 draw a random bird b1 from B
        # 24 draw (pb,1, sb,1) for bird where sb,1 is drawn from greater half of S
        # 25 resize d with respect to s2
        # 26 overlay f2 with d in position p2
        # 27 resize b1 with respect to sb,1
        # 28 overlay f1 with b1 in position pb,1
        # 29

        # 30 save f0, f1, f2 into the dataset
        # print(config)
        saved_configs.append(config)
        saved_config_counter+= 1
        # print(saved_config_counter, gen, s, d, b)
        # print(gen)
# 31 end

print("rejection probability", 1.0 - acceptance_ratio, "accept probability", acceptance_ratio)
print(saved_configs.__len__(), "of", total_configurations, "total configurations selected")

# TODO ndarry.choose??
