# Data Generator
The purpose of this python program is to generate training data for use with
opencv's Haar Cascade classifier. It implements a modified version of the algorithm
within the following paper.
 - Title:            Using Deep Networks for Drone Detection
 - Authors:          Aker, Cemal; Kalkan, Sinan
 - Publication:      eprint arXiv:1706.05726
 - Publication Date: 06/2017
 - Origin:           ARXIV
 - URL:              https://arxiv.org/abs/1706.05726

## Algorithm
The full modified algorithm is located in algorithm.md

## Usage
### Example Usage
```python3 drone_generator.py --max_configs 3000 --pos_dir drones --bg_dir bg_videos --neg_dir birds --out_dir dataset```
### Full Help Menu

```
usage: drone_generator.py [-h] [--min_size MIN_SIZE] [--max_size MAX_SIZE]
                          [--pos_dir POS_DIR] [--neg_dir NEG_DIR]
                          [--bg_dir BG_DIR] [--bg_width BG_WIDTH]
                          [--bg_height BG_HEIGHT] [--num_rows NUM_ROWS]
                          [--num_cols NUM_COLS] --max_configs MAX_CONFIGS
                          [--out_dir OUT_DIR] [--test_mode]

This python script generates object detection data-sets from positive example
objects, negative objects and background videos

optional arguments:
  -h, --help            show this help message and exit
  --min_size MIN_SIZE   Upper bound of sizing interval used to resize pasted
                        objects
  --max_size MAX_SIZE   Upper bound of sizing interval used to resize pasted
                        objects
  --pos_dir POS_DIR     Directory of source positive images for pasting.
  --neg_dir NEG_DIR     Directory of source negative images for pasting.
  --bg_dir BG_DIR       Directory of source background videos for pasting
                        onto.
  --bg_width BG_WIDTH   Background video width in pixels
  --bg_height BG_HEIGHT
                        Background video height in pixels
  --num_rows NUM_ROWS   Number of rows to split background image into for
                        positioning objects
  --num_cols NUM_COLS   Number of cols to split background image into for
                        positioning objects
  --max_configs MAX_CONFIGS
                        Maximum number of configurations to generate each
                        configuration creates 4 images
  --out_dir OUT_DIR     Output directory to save resulting data set
  --test_mode
```

# Background Remover
This is a helper script to remove backgrounds from images.
