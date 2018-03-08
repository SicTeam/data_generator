import itertools as it
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

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

# D ← foregrounds of drone images
drones_path = './' + 'drones' + '/'
drones_images = [drones_path + f for f in os.listdir(drones_path)]

# B ← foregrounds of bird images
birds_path = './' + 'birds' + '/'
birds_images = [birds_path + f for f in os.listdir(birds_path)]

# V ← background videos
bg_videos_path = './' + 'backgrounds' + '/'
bg_videos = [bg_videos_path + f for f in os.listdir(bg_videos_path)]
bg_video_dim = (800, 800)

# R ← # of rows that the image will be divided into
num_rows = 12

# C ← # of columns that the image will be divided into
num_cols = 10

# G ← R × C grid
# Building the grid outside of the main loop requires
# assuming the bg_video_dim is the same for all videos
# and known before data generation
grid_img_row = int(bg_video_dim[0] / num_rows)
grid_img_col = int(bg_video_dim[1] / num_cols)

grid = [(row, col) for row in range(0, bg_video_dim[0], grid_img_row) for col in
        range(0, bg_video_dim[1], grid_img_col)]

# plane_images = []  # Extra negative image examples

total_configurations = len(drones_images) * len(grid) * len(size_intervals) * len(bg_videos)
max_configurations = 5000

# Generator of possible configurations
configurations = it.product(drones_images, grid, size_intervals, bg_videos)

gen = 0
saved_config_cnt = 0
saved_configs = []
today = datetime.today()
output_dir = './output-' + today.strftime("%Y-%m-%d_%H:%M:%S") + '/'
pos_img_dir = 'pos_images/'
neg_img_dir = 'neg_images/'

try:
    os.mkdir(output_dir)  # TODO tidy these up
    os.mkdir(output_dir + pos_img_dir)
    os.mkdir(output_dir + neg_img_dir)  # needed for haar cascade specifically
except FileExistsError as e:
    print("Output folder '%s' already exists, exiting..." % e.filename)
    exit(1)

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


def get_pos_and_size(grid_position, size_interval):
    # Random Point x,y in g range(x, x+80), g range(y,y+80)
    x_pos = np.random.uniform(low=grid_position[1], high=min(grid_position[1] + grid_img_col, bg_video_dim[1]))
    y_pos = np.random.uniform(low=grid_position[0], high=min(grid_position[0] + grid_img_row, bg_video_dim[1]))
    size = np.random.uniform(low=size_interval[0], high=size_interval[1])
    return (int(x_pos), int(y_pos)), size


def paste_object(object_to_paste, background, position):
    # TODO pastes object outside of bounds of background image and moves pixels down.
    # either needs to skip that region of object or object position should be
    # based on object resize
    x_offset, y_offset = position[1], position[0]
    y1, y2 = y_offset, y_offset + object_to_paste.shape[0]
    x1, x2 = x_offset, x_offset + object_to_paste.shape[1]
    alpha_object = object_to_paste[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_object

    try:
        for c in range(0, 3):
            bg_region = background[y1:y2, x1:x2, c].shape  # background region to truncating alphas
            background[y1:y2, x1:x2, c] = (
                    alpha_object[:bg_region[0], :bg_region[1]] * object_to_paste[:bg_region[0], :bg_region[1], c] +
                    alpha_background[:bg_region[0], :bg_region[1]] * background[y1:y2, x1:x2, c])

            # if c == 0:
            #     print("truncated alpha object shape", alpha_object[:int(y2 - y1), :int(x2 - x1)].shape)
            #     print("truncated alpha background shape", alpha_background[:int(y2 - y1), :int(x2 - x1)].shape)
    except ValueError as err:
        err_str = "pos: (y,x) (%s) \n" \
                  "offset: (x, y) (%d,%d)\n" \
                  "paste range: \n" \
                  "\t xs in [x1:x2] [%d:%d]\n" \
                  "\t ys in [y1:y2] [%d:%d]\n" \
                  "alpha_object shape: %s\n" \
                  "alpha_background shape: %s\n" \
                  "object_to_paste shape: %s\n" \
                  "background shape: %s" % (
                      position, x_offset, y_offset, x1, x2, y1, y2, alpha_object.shape, alpha_background.shape,
                      object_to_paste.shape, background.shape)
        print(err_str)
        print(err, "bg shape: ", background[y1:y2, x1:x2, c].shape, y2 - y1, x2 - x1)
        print(alpha_object[:int(y2 - y1), :int(x2 - x1)].shape)
        print(alpha_background[:int(y2 - y1), :int(x2 - x1)].shape)
        pass

    return (position[1], position[0]), background, bg_region


def load_and_resize(obj_file_name, target_size):
    # Resize object with respect to size, size becomes smaller side of object
    obj = cv2.imread(obj_file_name, -1)  # -1 for unchanged - preserves alpha channel
    obj = cv2.cvtColor(obj, cv2.COLOR_BGRA2RGBA)
    source_size_smallest_edge = min(obj.shape[:2])
    scaled_size = target_size / source_size_smallest_edge

    if target_size < source_size_smallest_edge:
        obj = cv2.resize(obj, None, fx=scaled_size, fy=scaled_size, interpolation=cv2.INTER_LINEAR)
    else:
        obj = cv2.resize(obj, None, fx=scaled_size, fy=scaled_size, interpolation=cv2.INTER_AREA)
    return obj, obj.shape[0], obj.shape[1]


def save_frame(frame, number, config_number, location, shape, positive):
    x, y = location
    height, width = shape
    filename = str(config_number) + '-f' + str(number) + '.png'

    if positive:
        img_dir = pos_img_dir
        cv2.imwrite(output_dir + img_dir + filename, frame)
        return "%s%s 1 %s %s %s %s\n" % (img_dir, filename, x, y, width, height)
    else:
        img_dir = neg_img_dir
        cv2.imwrite(output_dir + img_dir + filename, frame)
        return "%s%s\n" % (img_dir, filename)


if __name__ == '__main__':

    t0 = time.time()  # Start Time
    pos_annotation_file = open(output_dir + "drone.txt", 'w')
    neg_annotation_file = open(output_dir + "bg.txt", 'w')

    for config in configurations:
        # foreach (d, g, s, v) ∈ D × G × S × V do

        # print(config)

        # 9 ignore this configuration with probability
        # p = 1 − Max. allowed size/ Total size for all configurations , and continue
        accept_config = np.random.choice(a=[False, True],
                                         p=[1.0 - acceptance_ratio, acceptance_ratio])

        gen += 1  # Generation counter

        if gen > 10:
            break

        if not accept_config:
            # print(gen, accept_config)
            continue
        else:
            # Unpack configuration : (d, g, s, v)
            # d - drone image filename
            # g - grid coordinates to place drone
            # s - size interval for resizing drone
            # v - background video filename
            d, g, s, v = config

            # draw a random position p0 in g
            # draw a random size s0 for smaller edge of the drone from s
            p0, s0 = get_pos_and_size(g, s)

            # draw a random frame f0 from v
            _, _, _, f0 = frame_grabber(v)

            # resize d with respect to s0
            d0, drone_w, drone_h = load_and_resize(d, s0)

            # overlay f0 with d in position p0
            d0_loc, f0, d0_shape = paste_object(d0, f0, p0)

            # draw (p1, s1, f1) in the same way
            p1, s1 = get_pos_and_size(g, s)
            _, _, _, f1 = frame_grabber(v)

            # resize d with respect to s1
            d1, d1width, d1height = load_and_resize(d, s1)

            # overlay f1 with d in position p1
            d1_loc, f1, d1_shape = paste_object(d1, f1, p1)

            # draw (p2, s2, f2) in the same way
            p2, s2 = get_pos_and_size(g, s)
            _, _, _, f2 = frame_grabber(v)

            # resize d with respect to s2
            d2, d2width, d2height = load_and_resize(d, s2)

            # overlay f2 with d in position p2
            d2_loc, f2, d2_shape = paste_object(d2, f2, p2)

            # split size interval list at middle
            sz_mid_idx = int(len(size_intervals) / 2) + 1
            sb_lower = size_intervals[:sz_mid_idx][np.random.randint(low=0, high=sz_mid_idx)]
            sb_upper = size_intervals[sz_mid_idx:][np.random.randint(low=0, high=len(size_intervals) - sz_mid_idx)]

            # draw a random bird b0 from B and a random position
            b0 = np.random.choice(birds_images)

            # draw (pb,0, sb,0) for bird
            #   where pb,0 is a random grid position from grid list
            #   where sb,0 is drawn from smaller half of S
            gb0 = grid[np.random.randint(low=0, high=len(grid))]
            pb0, sb0 = get_pos_and_size(gb0, sb_lower)

            # resize b0 with respect to sb,0
            bird0, _, _ = load_and_resize(b0, sb0)

            # draw a random bird b1 from B and random position
            b1 = np.random.choice(birds_images)
            gb1 = grid[np.random.randint(low=0, high=len(grid))]

            # draw (pb,1, sb,1) for bird
            #   where pb,1 is a random grid position from grid list
            #   where sb,1 is drawn from greater half of S
            pb1, sb1 = get_pos_and_size(gb1, sb_upper)

            # resize b1 with respect to sb,1
            bird1, _, _ = load_and_resize(b1, sb1)

            # overlay f1 with b0 in position pb,0
            _, f1, _ = paste_object(bird0, f1, pb0)

            extra_bird = np.random.choice([True, False])
            if extra_bird:
                # overlay f1 with b1 in position pb,1
                _, f1, _ = paste_object(bird1, f1, pb1)

            # grab random frame from v f4
            _, _, _, f3 = frame_grabber(v)

            # overlay f4 with b0 in position pb,0
            _, f3, _ = paste_object(bird0, f3, pb0)

            extra_bird = np.random.choice([True, False])
            if extra_bird:
                # overlay f3 with b1 in position pb,1
                _, f3, _ = paste_object(bird1, f3, pb1)

            # save f0, f1, f2, f3 into the data set

            saved_configs.append(config)
            saved_config_cnt += 1

            annotation_str = save_frame(f0, 0, saved_config_cnt, d0_loc, d0_shape, positive=True)
            annotation_str += save_frame(f1, 1, saved_config_cnt, d1_loc, d1_shape, positive=True)
            annotation_str += save_frame(f2, 2, saved_config_cnt, d2_loc, d2_shape, positive=True)
            # save to data description text positive drone images must follow this format
            # "path/to/file.png 1 197 59 223 162" : boxcount, x, y, width, height
            pos_annotation_file.write(annotation_str)

            neg_annotation_file.write(save_frame(f3, 3, saved_config_cnt, _, _, positive=False))

            # im0.save(out_file_name)
            # cv2.imwrite(out_file_name, f0)

            # print(b0, b1)

            # print(saved_config_counter, gen, s, d, b)
            # print(gen)
            # 31 end

    pos_annotation_file.close()
    neg_annotation_file.close()
    t1 = time.time()  # End Time

    run_time = t1 - t0

    rt_hour, rt_min, rt_sec = (run_time / 3600), (run_time % 3600) / 60, (run_time % 3600) % 60

    saved_pos_img_count = len(os.listdir(output_dir + pos_img_dir))
    saved_neg_img_count = len(os.listdir(output_dir + neg_img_dir))

    print("Total Run Time (h:m:s) : %i:%i:%i\n"
          "Configurations:\n"
          "\tProcessed = %i\n"
          "\tAccepted = %i\n"
          "\tTotal Possible = %i\n"
          "\tReject Probability = %f\n"
          "\tAccept Probability = %f\n"
          "Images: \n"
          "\tSaved Pos. = %i\n"
          "\tSaved Neg. = %i\n"
          "\tObject Positives = %i\n"
          "\tObject Negatives = %i\n"
          "\tBackground Videos = %i\n"
          % (rt_hour, rt_min, rt_sec
             , gen
             , len(saved_configs)
             , total_configurations
             , 1.0 - acceptance_ratio
             , acceptance_ratio
             , saved_pos_img_count
             , saved_neg_img_count
             , len(drones_images)
             , len(birds_images)
             , len(bg_videos)
             )
          )
