import itertools as it

import numpy as np

# Algorithm 1: The algorithm for preparing the dataset.
# 1 S ← predefined size intervals
# TODO numpy choice with probability
# TODO how to skew results towards lower numbers
sizes = np.random.uniform(5, 160, 20).reshape(10, 2)

# 2 D ← foregrounds of drone images
drones_images = ["drone3", "drone2", "drone1"]

# 3 B ← foregrounds of bird images
bird_images = ["bird1", "bird2", "bird3"]

# 4 V ← background videos
bg_video = ["vid1", "vid2", "vid3"]

# 5 R ← # of rows that the image will be divided into
num_rows = 12

# 6 C ← # of columns that the image will be divided into
num_cols = 10

# 7 G ← R × C grid
grid = [(row, col) for row in range(num_rows) for col in range(num_cols)]

plane_images = []  # Extra negative image examples

total_configurations = len(sizes) * len(drones_images) * len(bird_images) * len(bg_video) * len(grid)
max_configurations = 1000

# Generator of possible configurations
configurations = it.product(sizes, drones_images, bird_images, bg_video, grid)

gen = 0
saved_configs = []

acceptance_ratio = float(max_configurations / total_configurations)
for config in configurations:
    # 8 foreach (d, g, s, v) ∈ D × G × S × V do

    # print(config)

    # 9 ignore this configuration with probability
    # p = 1 − Max. allowed size/ Total size for all configurations , and continue
    accept_config = np.random.choice(a=[False, True],
                                     p=[1.0 - acceptance_ratio, acceptance_ratio])

    gen += 1  # Generation counter

    if not accept_config:
        print(gen, accept_config)
    else:
        # TODO this shouldn't be random? why? does it matter?
        d = np.random.choice(drones_images)

        b = np.random.choice(bird_images)

        s = sizes[np.random.random_integers(0, 9)]  # TODO choose at random from pre defined intervals
        # currently picking random index.

        # print(gen, accept_config, d, b, s, max_configurations, total_configurations)
        print(gen, accept_config, config)
        # 10 draw a random position p0 in g
        # 11 draw a random size s0 for smaller edge of the
        # drone from s
        # 12 draw a random frame f0 from v
        # 13 resize d with respect to s0
        # 14 overlay f0 with d in position p0
        # 15 draw (p1, s1, f1) in the same way
        # 16 draw a random bird b0 from B
        # 17 draw (pb,0, sb,0) for bird where sb,0 is drawn from
        # smaller half of S
        # 18 resize d with respect to s1
        # 19 overlay f1 with d in position p1
        # 20 resize b0 with respect to sb,0
        # 21 overlay f1 with b0 in position pb,0
        # 22 draw (p2, s2, f2) in the same way
        # 23 draw a random bird b1 from B
        # 24 draw (pb,1, sb,1) for bird where sb,1 is drawn from
        # greater half of S
        # 25 resize d with respect to s2
        # 26 overlay f2 with d in position p2
        # 27 resize b1 with respect to sb,1
        # 28 overlay f1 with b1 in position pb,1
        # 29

        # 30 save f0, f1, f2 into the dataset
        saved_configs.append(config)
# 31 end

print("rejection probability", 1.0 - acceptance_ratio, "accept probability", acceptance_ratio)
print(saved_configs.__len__())

# TODO ndarry.choose??
