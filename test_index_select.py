import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np

A = [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]], [[13,14,15], [16,17,18]]]
A = np.array(A)

A = torch.from_numpy(A)

idx = [i for i in range(A.size(2) - 1, -1, -1)]
idx = torch.LongTensor(idx)
B = A.index_select(2, idx)

print('OK')