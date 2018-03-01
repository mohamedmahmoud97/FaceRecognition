import os
from math import sqrt, ceil
from PIL import Image
import numpy as np
def walk_dir(root_str, action_at_leaf, action_at_parent):

    directory = os.fsencode(root_str)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        file_path = os.path.join(root_str, filename)
        if os.path.isdir(os.fsencode(file_path)):
            action_at_parent(file_path)
            walk_dir(file_path,action_at_leaf, action_at_parent)
        elif os.path.isfile(file_path):
            action_at_leaf(file_path)
            

def load_img_into(X):
    def load_img(path):
        img = Image.open(path)
        X.append(np.array(img))
    return load_img

def set_label_into(y):
    def set_label(path):
        y.extend([int(os.path.basename(path).replace('s',''))]*10)
    return set_label

def load_faces(root_path):
    X = []
    y = []
    walk_dir(root_path, load_img_into(X), set_label_into(y))
    y = np.array(y)
    X = np.array(X).reshape(400,-1)
    return X, y

def visualize_grid(Xs, ubound=255.0, padding=1):
    (N, H, W) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width))
    next_idx = 0
    y0, y1 = 0, H
    for _ in range(grid_size):
        x0, x1 = 0, W
        for _ in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding

    return grid

