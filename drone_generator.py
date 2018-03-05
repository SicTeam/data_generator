import itertools as it
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image

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
drones_path = './' + 'drones' + '/'
# drones_images = os.listdir(drones_path)
drones_images = [drones_path + f for f in os.listdir(drones_path)]

# 3 B ← foregrounds of bird images
birds_path = './' + 'birds' + '/'
# bird_images = os.listdir('./birds')
birds_images = [birds_path + f for f in os.listdir(birds_path)]

# 4 V ← background videos
bg_videos_path = './' + 'backgrounds' + '/'
# bg_videos = os.listdir('./backgrounds')
bg_videos = [bg_videos_path + f for f in os.listdir(bg_videos_path)]

bg_video_dim = (800, 800)

# 5 R ← # of rows that the image will be divided into
num_rows = 12

# 6 C ← # of columns that the image will be divided into
num_cols = 10

# 7 G ← R × C grid

# Building the grid outside of the main loop requires
# assuming the bg_video_dim is the same for all videos
# and known before data generation
grid_img_row = int(bg_video_dim[0] / num_rows)
grid_img_col = int(bg_video_dim[1] / num_cols)

grid = [(row, col) for row in range(0, bg_video_dim[0], grid_img_row) for col in
        range(0, bg_video_dim[1], grid_img_col)]

# plane_images = []  # Extra negative image examples

total_configurations = len(drones_images) * len(grid) * len(size_intervals) * len(bg_videos)
max_configurations = 500

# Generator of possible configurations
configurations = it.product(drones_images, grid, size_intervals, bg_videos)

gen = 0
saved_config_counter = 0
saved_configs = []

acceptance_ratio = float(max_configurations / total_configurations)


def frame_grabber(src_video):
    bg = cv2.VideoCapture(src_video)
    fps = bg.get(cv2.CAP_PROP_FPS)
    frame_count = bg.get(cv2.CAP_PROP_FRAME_COUNT)
    width = bg.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = bg.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_number = np.random.randint(frame_count)
    bg.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, cv2_im = bg.read()
    # cv2_im = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    bg.release()
    print("fps: %f, frames: %d, width: %d, height: %d " % (fps, frame_count, width, height))
    return width, height, frame_number, cv2_im


def get_pos_and_size(g, s):
    # TODO RANDOM POINT x,y = g range(x, x+80), range(y,y+80)
    x_pos = np.random.uniform(low=g[1], high=g[1] + grid_img_col)
    y_pos = np.random.uniform(low=g[0], high=g[0] + grid_img_row)
    size = np.random.uniform(low=s[0], high=s[1])
    return (x_pos, y_pos), size


def paste_object(object, background, position):
    x_offset, y_offset = position
    y1, y2 = y_offset, y_offset + object.shape[0]
    x1, x2 = x_offset, x_offset + object.shape[1]
    alpha_object = object[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_object
    for c in range(0, 3):
        background[y1:y2, x1:x2, c] = (alpha_object * object[:, :, c] +
                                       alpha_background * background[y1:y2, x1:x2, c])
    return position, background


for config in configurations:
    # 8 foreach (d, g, s, v) ∈ D × G × S × V do

    # print(config)

    # 9 ignore this configuration with probability
    # p = 1 − Max. allowed size/ Total size for all configurations , and continue
    accept_config = np.random.choice(a=[False, True],
                                     p=[1.0 - acceptance_ratio, acceptance_ratio])

    gen += 1  # Generation counter

    if gen > 10000:
        break

    if not accept_config:
        # print(gen, accept_config)
        continue
    else:

        # Unpack configuration : (d, g, s, v)
        d, g, s, v = config

        b = np.random.choice(birds_images)

        print(d, g, s, v)

        # 10 draw a random position p0 in g
        # 11 draw a random size s0 for smaller edge of the drone from s
        p0, s0 = get_pos_and_size(g, s)  # save this for bounding box of drone

        # 12 draw a random frame f0 from v
        _, _, _, f0 = frame_grabber(v)

        # 13 resize d with respect to s0
        # TODO make this a function to resize based on s0 also choose interpolation based on growing or shrinking
        drone = cv2.imread(d, -1)  # -1 for unchanged - preserves alpha channel
        drone = cv2.resize(drone, None, fx=.125, fy=.125, interpolation=cv2.INTER_AREA)

        # 14 overlay f0 with d in position p0
        drone_location, f0 = paste_object(drone, f0, p0)

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
        # Protects against overflow of images
        # TODO make this a debug only option
        if saved_config_counter < 5:
            im = Image.fromarray(f0)
            im.show()

        saved_configs.append(config)
        saved_config_counter += 1
        # print(saved_config_counter, gen, s, d, b)
        # print(gen)
# 31 end
# Image.fromarray(f0).imshow()
print("rejection probability", 1.0 - acceptance_ratio, "accept probability", acceptance_ratio)
print(saved_configs.__len__(), "of", total_configurations, "total configurations selected")

# TODO ndarry.choose??
