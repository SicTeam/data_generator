"""
The MIT License

Copyright (c) 2018 SicTeam - (Portland State University)
SicTeam is: Israel Bond, Brandon Craig, Cody Herberholz, Khuong Nguyen,
            Dakota Sanchez, Samuel Strba, Elijah Whitham-Powell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

The purpose of this python program is to generate training data for use with
opencv's Haar Cascade classifier. It implements a modified version of the algorithm
within the following paper.
      Title:            Using Deep Networks for Drone Detection
      Authors:          Aker, Cemal
                        Kalkan, Sinan
      Publication:      eprint arXiv:1706.05726
      Publication Date: 06/2017
      Origin:           ARXIV
      URL:              https://arxiv.org/abs/1706.05726

The modified algorithm is as follows,
Algorithm # 1: The algorithm for preparing the data-set:
     1 S ← predefined size intervals\n
     2 D ← foregrounds of drone images\n
     3 B ← foregrounds of bird images\n
     4 V ← background videos\n
     5 R ←  # of rows that the image will be divided into\n
     6 C ←  # of columns that the image will be divided into\n
     7 G ← R × C grid\n
     8 foreach(d, g, s, v) ∈ D × G × S × V do\n
     9    ignore this configuration with probability
             p = 1 − Max.allowed size / Total size for all configurations
          and continue\n
    10    draw a random position p0 in g\n
    11    draw a random size s0 for smaller edge of the drone from s\n
    12    draw a random frame f0 from v\n
    13    resize d with respect to s0\n
    14    overlay f0 with d in position p0\n
    15    draw (p1, s1, f1) in the same way\n
    16    resize d with respect to s1\n
    17    overlay f1 with d in position p1\n
    18    draw (p2, s2, f2) in the same way\n
    19    resize d with respect to s2\n
    19    overlay f2 with d in position p2\n
    20    split size interval list at middle\n
    21    draw a random bird b0 from B and a random position\n
    22    draw (pb,0, sb,0) for bird b0\n
             where pb,0 is a random grid position from grid list\n
             where sb,0 is drawn from smaller half of S\n
    23    resize b0 with respect to sb,0\n
    24    draw a random bird b1 from B and random position\n
    25    draw (pb,1, sb,1) for bird b1\n
               where pb,1 is a random grid position from grid list\n
              where sb,1 is drawn from greater half of S\n
    26    resize b1 with respect to sb,1\n
    27    overlay f1 with b0 in position pb,0\n
    28    randomly decide to overlay a second bird\n
    29    overlay f1 with b1 in position pb,1\n
    30    grab random frame from v f3\n
    31    overlay f3 with b0 in position pb,0\n
    32    randomly choose to add a second bird\n
    33    overlay f3 with b1 in position pb,1\n
    34    save f0, f1, f2, f3 into the data set\n
    35 End Loop\n

"""
import argparse
import itertools as it
import os
import sys
import time
from datetime import datetime

import cv2  # opencv
import numpy as np
import pandas as pd


def frame_grabber(src_video: str):
    """
    Grabs and returns a random frame from a provided video.

    Args:
        src_video: str : path to file

    Returns:
        width: pixels : int
        height: pixels : int
        frame_number: the frame number selected from src_video : int
        cv2_im: the frame as an array-like object
    """
    bg = cv2.VideoCapture(src_video)
    # fps = bg.get(cv2.CAP_PROP_FPS)
    frame_count = bg.get(cv2.CAP_PROP_FRAME_COUNT)
    width = bg.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = bg.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_number = np.random.randint(frame_count)
    bg.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    _, cv2_im = bg.read()
    bg.release()
    return width, height, frame_number, cv2_im


def paste_object(object_to_paste, background, position):
    """
    Pastes and object onto a background image at a given position.


    Args:
        object_to_paste: array-like : shape { height, width, channels }
        background: array-like : shape { height, width, channels }
        position: tuple :  (row, col)

    Returns:
        (position[1], position[0]): tuple, (x,y) - top corner of pasted object
        background: array-like, shape { height, width, channels } - resulting image
        bg_region: tuple, (height, width) - pasted region for annotation

    """
    x_offset, y_offset = position[1], position[0]
    y1, y2 = y_offset, y_offset + object_to_paste.shape[0]
    x1, x2 = x_offset, x_offset + object_to_paste.shape[1]
    alpha_object = object_to_paste[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_object
    c = None
    bg_region = None

    try:
        for c in range(0, 3):
            bg_region = background[y1:y2, x1:x2, c].shape  # background region to truncating alphas
            background[y1:y2, x1:x2, c] = (
                    alpha_object[:bg_region[0], :bg_region[1]] * object_to_paste[:bg_region[0], :bg_region[1], c] +
                    alpha_background[:bg_region[0], :bg_region[1]] * background[y1:y2, x1:x2, c])

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
    """
    Loads an image from file path and re-sizes its smallest edge to the size
    provided.
    Args:
        obj_file_name: str, path to image file
        target_size: float or int, target size of smallest edge

    Returns:
        obj: array-like, shape { new width, new height }
        obj.shape[0]: int, width
        obj.shape[1]: int, height
    """
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


class Builder:
    """
        This class implements the following algorithm to generate training images for drone detection.
        The output is a set of positive images and a set of negative images with corresponding
        annotation files formatted for opencv_traincascade Haar Cascade classifier. The algorithm
        was modified from a data generation algorithm specified in,
            Title:    Using Deep Networks for Drone Detection
            Authors:  Aker, Cemal
                      Kalkan, Sinan
            Publication: eprint arXiv:1706.05726
            Publication Date: 06/2017
            Origin:	ARXIV
            URL: https://arxiv.org/abs/1706.05726

        Algorithm # 1: The algorithm for preparing the data-set:
              1 S ← predefined size intervals\n
              2 D ← foregrounds of drone images\n
              3 B ← foregrounds of bird images\n
              4 V ← background videos\n
              5 R ←  # of rows that the image will be divided into\n
              6 C ←  # of columns that the image will be divided into\n
              7 G ← R × C grid\n
              8 foreach(d, g, s, v) ∈ D × G × S × V do\n
              9    ignore this configuration with probability
                       p = 1 − Max.allowed size / Total size for all configurations
                   and continue\n
             10    draw a random position p0 in g\n
             11    draw a random size s0 for smaller edge of the drone from s\n
             12    draw a random frame f0 from v\n
             13    resize d with respect to s0\n
             14    overlay f0 with d in position p0\n
             15    draw (p1, s1, f1) in the same way\n
             16    resize d with respect to s1\n
             17    overlay f1 with d in position p1\n
             18    draw (p2, s2, f2) in the same way\n
             19    resize d with respect to s2\n
             19    overlay f2 with d in position p2\n
             20    split size interval list at middle\n
             21    draw a random bird b0 from B and a random position\n
             22    draw (pb,0, sb,0) for bird b0\n
                       where pb,0 is a random grid position from grid list\n
                       where sb,0 is drawn from smaller half of S\n
             23    resize b0 with respect to sb,0\n
             24    draw a random bird b1 from B and random position\n
             25    draw (pb,1, sb,1) for bird b1\n
                       where pb,1 is a random grid position from grid list\n
                       where sb,1 is drawn from greater half of S\n
             26    resize b1 with respect to sb,1\n
             27    overlay f1 with b0 in position pb,0\n
             28    randomly decide to overlay a second bird\n
             29    overlay f1 with b1 in position pb,1\n
             30    grab random frame from v f3\n
             31    overlay f3 with b0 in position pb,0\n
             32    randomly choose to add a second bird\n
             33    overlay f3 with b1 in position pb,1\n
             34    save f0, f1, f2, f3 into the data set\n
             35 End Loop\n
    """

    def __init__(self,
                 max_configs,
                 sz_lower_bound=5,
                 sz_upper_bound=160,
                 drones_path='./drones/',
                 birds_path='./birds/',
                 bg_videos_path='./backgrounds/',
                 bg_video_dim=(800, 800),
                 num_rows=12,
                 num_cols=10,
                 output_dir='./output-'):
        """
        Initializes the data builder object for data generation.

        Args:
            max_configs: int, maximum number of configurations to save - NOTE: each configuration generates 4 images (3 pos, 1 negative)
            sz_lower_bound: int, lower bound for sizing intervals
            sz_upper_bound: int, upper bound for sizing intervals
            drones_path: str, path to positive objects with backgrounds removed
            birds_path: str, path to negative objects with backgrounds removed
            bg_videos_path: str, path to background videos
            bg_video_dim: (int, int), shape of background videos in pixels (height, width)
            num_rows: int, number of rows to split background into for pasting
            num_cols: int, number of column to split background into for pasting
            output_dir: str, path to dir where output is saved.
        """

        # S ← predefined size intervals
        self.sz_lower_bound = sz_lower_bound
        self.sz_upper_bound = sz_upper_bound
        self.sz_interval_length = self.sz_upper_bound - self.sz_lower_bound
        self.sz_interval_midpoint = int(self.sz_interval_length / 2)
        # Generate uniform intervals within the given upper and lower bounds with more in the lower half than in the upper half

        bins = np.concatenate((pd.cut(np.arange(self.sz_lower_bound, self.sz_interval_midpoint), 14, retbins=True)[1],
                               pd.cut(np.arange(self.sz_interval_midpoint, self.sz_upper_bound), 5, retbins=True)[1]))
        self.size_intervals = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

        # D ← foregrounds of drone images
        self.pos_path = drones_path
        self.drone_images = [self.pos_path + f for f in os.listdir(self.pos_path)]

        # B ← foregrounds of bird images
        self.birds_path = birds_path
        self.bird_images = [self.birds_path + f for f in os.listdir(self.birds_path)]

        # V ← background videos
        self.bg_videos_path = bg_videos_path
        self.bg_videos = [self.bg_videos_path + f for f in os.listdir(self.bg_videos_path)]

        # R ← # of rows that the image will be divided into
        self.num_rows = num_rows

        # C ← # of columns that the image will be divided into
        self.num_cols = num_cols

        # G ← R × C grid
        # Building the grid outside of the main loop requires
        # assuming the bg_video_dim is the same for all videos
        # and known before data generation
        self.bg_dim = bg_video_dim
        self.grid_img_row = int(self.bg_dim[0] / self.num_rows)
        self.grid_img_col = int(self.bg_dim[1] / self.num_cols)
        self.grid = [(row, col) for row in range(0, self.bg_dim[0], self.grid_img_row) for col in
                     range(0, self.bg_dim[1], self.grid_img_col)]

        # Generator of possible configurations
        self.configurations = it.product(self.drone_images, self.grid, self.size_intervals, self.bg_videos)

        self.gen = 0  # generated configuration counter
        self.saved_config_cnt = 0
        self.saved_configs = []
        today = datetime.today()
        self.pos_img_dir = 'pos_images/'
        self.neg_img_dir = 'neg_images/'
        self.output_dir = output_dir + today.strftime("%Y-%m-%d_%H:%M:%S") + '/'
        try:
            os.mkdir(self.output_dir)  # TODO tidy these up
            os.mkdir(self.output_dir + self.pos_img_dir)
            os.mkdir(self.output_dir + self.neg_img_dir)  # needed for haar cascade specifically
        except FileExistsError as e:
            print("Output folder '%s' already exists, exiting..." % e.filename)
            exit(1)

        self.max_configurations = max_configs
        self.total_configurations = len(self.drone_images) * len(self.grid) * len(self.size_intervals) * len(
            self.bg_videos_path)
        self.acceptance_ratio = float(self.max_configurations / self.total_configurations)

    def get_pos_and_size(self, grid_position, size_interval):
        """
        Get a random position from within the grid coordinates provided

        Args:
            grid_position: (int, int), the grid location to select exact paste coordinates (col, row)
            size_interval: (float, float), size interval from which to select a random size.

        Returns:
            (x_pos, y_pos): (int, int), exact random x,y position from within the selected grid position.
            size: float, randomly selected size from size interval.

        """
        # Random Point x,y in g range(x, x+80), g range(y,y+80)
        x_pos = np.random.uniform(low=grid_position[1], high=min(grid_position[1] + self.grid_img_col, self.bg_dim[1]))
        y_pos = np.random.uniform(low=grid_position[0], high=min(grid_position[0] + self.grid_img_row, self.bg_dim[1]))
        size = np.random.uniform(low=size_interval[0], high=size_interval[1])
        return (int(x_pos), int(y_pos)), size

    def save_frame(self, frame, number, config_number, location, shape, positive):
        """
        Saves a given frame to file.

        Args:
            frame: array-like, shape { rows, cols, channels }
            number: int, frame number for naming (0-3)
            config_number: int, saved configuration number for naming
            location: (int, int), top right corner of bounding box for pasted object (x, y)
            shape: (int, int), height and width of the bounding box from the top right corner (height, width)
            positive: bool, indicates if the file should be saved as a positive or negative example image

        Returns:

        """
        x, y = location
        height, width = shape
        filename = str(config_number) + '-f' + str(number) + '.png'

        if positive:
            img_dir = self.pos_img_dir
            cv2.imwrite(self.output_dir + img_dir + filename, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])
            return "%s%s 1 %s %s %s %s\n" % (img_dir, filename, x, y, width, height)
        else:
            img_dir = self.neg_img_dir
            cv2.imwrite(self.output_dir + img_dir + filename, frame, params=[cv2.IMWRITE_PNG_COMPRESSION, 9])
            return "%s%s\n" % (img_dir, filename)

    def run(self, testing_mode=False, testing_mode_generation_limit=10):
        """

        Args:
            testing_mode:
            testing_mode_generation_limit:
        """
        t0 = time.time()  # Start Time
        pos_annotation_file = open(self.output_dir + "pos.txt", 'w')
        neg_annotation_file = open(self.output_dir + "bg.txt", 'w')
        for config in self.configurations:

            self.gen += 1  # Generation counter

            if testing_mode:
                accept_config = True
            else:
                # Ignore this configuration with probability
                # p = 1 - Max. allowed size / Total size for all configurations , and continue
                accept_config = np.random.choice(a=[False, True],
                                                 p=[1.0 - self.acceptance_ratio, self.acceptance_ratio])

            if self.gen > testing_mode_generation_limit and testing_mode:
                break

            if self.saved_config_cnt >= self.max_configurations:
                break

            if not accept_config:
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
                p0, s0 = self.get_pos_and_size(g, s)

                # draw a random frame f0 from v
                _, _, _, f0 = frame_grabber(v)

                # resize d with respect to s0
                d0, drone_w, drone_h = load_and_resize(d, s0)

                # overlay f0 with d in position p0
                d0_loc, f0, d0_shape = paste_object(d0, f0, p0)

                # draw (p1, s1, f1) in the same way
                p1, s1 = self.get_pos_and_size(g, s)
                _, _, _, f1 = frame_grabber(v)

                # resize d with respect to s1
                d1, d1width, d1height = load_and_resize(d, s1)

                # overlay f1 with d in position p1
                d1_loc, f1, d1_shape = paste_object(d1, f1, p1)

                # draw (p2, s2, f2) in the same way
                p2, s2 = self.get_pos_and_size(g, s)
                _, _, _, f2 = frame_grabber(v)

                # resize d with respect to s2
                d2, d2width, d2height = load_and_resize(d, s2)

                # overlay f2 with d in position p2
                d2_loc, f2, d2_shape = paste_object(d2, f2, p2)

                # split size interval list at middle
                sz_mid_idx = int(len(self.size_intervals) / 2) + 1
                sb_lower = self.size_intervals[:sz_mid_idx][np.random.randint(low=0, high=sz_mid_idx)]
                sb_upper = self.size_intervals[sz_mid_idx:][
                    np.random.randint(low=0, high=len(self.size_intervals) - sz_mid_idx)]

                # draw a random bird b0 from B and a random position
                b0 = np.random.choice(self.bird_images)

                # draw (pb,0, sb,0) for bird b0
                #   where pb,0 is a random grid position from grid list
                #   where sb,0 is drawn from smaller half of S
                gb0 = self.grid[np.random.randint(low=0, high=len(self.grid))]
                pb0, sb0 = self.get_pos_and_size(gb0, sb_lower)

                # resize b0 with respect to sb,0
                bird0, _, _ = load_and_resize(b0, sb0)

                # draw a random bird b1 from B and random position
                b1 = np.random.choice(self.bird_images)
                gb1 = self.grid[np.random.randint(low=0, high=len(self.grid))]

                # draw (pb,1, sb,1) for bird b1
                #   where pb,1 is a random grid position from grid list
                #   where sb,1 is drawn from greater half of S
                pb1, sb1 = self.get_pos_and_size(gb1, sb_upper)

                # resize b1 with respect to sb,1
                bird1, _, _ = load_and_resize(b1, sb1)

                # overlay f1 with b0 in position pb,0
                _, f1, _ = paste_object(bird0, f1, pb0)

                extra_bird = np.random.choice([True, False])
                if extra_bird:
                    # overlay f1 with b1 in position pb,1
                    _, f1, _ = paste_object(bird1, f1, pb1)

                # grab random frame from v f3
                _, _, _, f3 = frame_grabber(v)

                # overlay f3 with b0 in position pb,0
                _, f3, _ = paste_object(bird0, f3, pb0)

                extra_bird = np.random.choice([True, False])
                if extra_bird:
                    # overlay f3 with b1 in position pb,1
                    _, f3, _ = paste_object(bird1, f3, pb1)

                # save f0, f1, f2, f3 into the data set
                self.saved_configs.append(config)
                self.saved_config_cnt += 1

                pos_annotation_str = self.save_frame(f0, 0, self.saved_config_cnt, d0_loc, d0_shape, positive=True)
                pos_annotation_str += self.save_frame(f1, 1, self.saved_config_cnt, d1_loc, d1_shape, positive=True)
                pos_annotation_str += self.save_frame(f2, 2, self.saved_config_cnt, d2_loc, d2_shape, positive=True)
                # save to data description text positive drone images must follow this format
                # "path/to/file.png 1 197 59 223 162" : box-count, x, y, width, height
                pos_annotation_file.write(pos_annotation_str)

                neg_annotation_file.write(self.save_frame(f3, 3, self.saved_config_cnt, _, _, positive=False))

                print("Processed: %d\t Accepted: %d \t Max Saved Configs: %d "
                      % (self.gen, self.saved_config_cnt, self.max_configurations))
        pos_annotation_file.close()
        neg_annotation_file.close()
        t1 = time.time()  # End Time
        run_time = t1 - t0
        rt_hour, rt_min, rt_sec = (run_time / 3600), (run_time % 3600) / 60, (run_time % 3600) % 60
        saved_pos_img_count = len(os.listdir(self.output_dir + self.pos_img_dir))
        saved_neg_img_count = len(os.listdir(self.output_dir + self.neg_img_dir))
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
                 , self.gen
                 , len(self.saved_configs)
                 , self.total_configurations
                 , 1.0 - self.acceptance_ratio
                 , self.acceptance_ratio
                 , saved_pos_img_count
                 , saved_neg_img_count
                 , len(self.drone_images)
                 , len(self.bird_images)
                 , len(self.bg_videos)
                 )
              )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This python script generates object detection "
                                                 "data-sets from positive example objects, negative objects"
                                                 " and background videos")
    # size interval
    parser.add_argument('--min_size',
                        default=5,
                        type=int,
                        help='Upper bound of sizing interval used to resize pasted objects')

    parser.add_argument('--max_size',
                        default=160,
                        type=int
                        , help='Upper bound of sizing interval used to resize pasted objects')

    # pos img src dir
    parser.add_argument('--pos_dir',
                        default='./drones/',
                        help='Directory of source positive images for pasting.')

    # neg img src dir
    parser.add_argument('--neg_dir',
                        default='./birds/',
                        help='Directory of source negative images for pasting.')
    # bg video src dir
    parser.add_argument('--bg_dir',
                        default='./backgrounds/',
                        help='Directory of source background videos for pasting onto.')

    # bg video dimensions
    parser.add_argument('--bg_width',
                        default='800',
                        type=int,
                        help='Background video width in pixels')

    parser.add_argument('--bg_height',
                        default='800',
                        type=int,
                        help='Background video height in pixels')

    # number of rows
    parser.add_argument('--num_rows',
                        default='12',
                        type=int,
                        help='Number of rows to split background image into for positioning objects')

    # number of columns
    parser.add_argument('--num_cols',
                        default='10',
                        type=int,
                        help='Number of cols to split background image into for positioning objects')

    # max configurations
    parser.add_argument('--max_configs',
                        required=True,
                        type=int,
                        help='Maximum number of configurations to generate each configuration creates 4 images')

    # output dir
    parser.add_argument('--out_dir',
                        default='./output-',
                        help='Output directory to save resulting data set')

    # test mode
    parser.add_argument('--test_mode',
                        action='store_true')

    # get arguments
    args = parser.parse_args(sys.argv[1:])

    # make builder object
    builder = Builder(max_configs=args.max_configs,
                      sz_lower_bound=args.min_size,
                      sz_upper_bound=args.max_size,
                      drones_path=args.pos_dir,
                      birds_path=args.neg_dir,
                      bg_videos_path=args.bg_dir,
                      bg_video_dim=(args.bg_height, args.bg_width),
                      num_rows=args.num_rows,
                      num_cols=args.num_cols,
                      output_dir=args.out_dir)

    # execute builder algorithm
    builder.run(testing_mode=args.test_mode)
