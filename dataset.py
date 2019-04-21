# -*- coding: utf-8 -*-
import os
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Resize


def is_image_file(filename):
	return any (filename.endswith (extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])


def load_img(filepath):
	y = Image.open (filepath).convert ('L')
	return y


class DatasetFromFolder (data.Dataset):
	def __init__(self, LR_image_dir, HR_image_dir, LR_transform=None, HR_transform=None):
		super (DatasetFromFolder, self).__init__ ()
		self.LR_image_filenames = sorted (
			[os.path.join (LR_image_dir, x) for x in os.listdir (LR_image_dir) if is_image_file (x)])
		self.HR_image_filenames = sorted (
			[os.path.join (HR_image_dir, y) for y in os.listdir (HR_image_dir) if is_image_file (y)])
		self.LR_transform = LR_transform
		self.HR_transform = HR_transform

	def __getitem__(self, index):
		inputs = load_img (self.LR_image_filenames[index])
		labels = load_img (self.HR_image_filenames[index])
		HR_images = self.HR_transform (labels)
		LR_images = self.LR_transform (inputs)
		return LR_images, HR_images

	def __len__(self):
		return len (self.LR_image_filenames)


def LR_transform(img_height, img_width, scale_factor):
	return Compose ([Resize ((img_height // scale_factor, img_width // scale_factor), Image.BICUBIC), ToTensor ()])


def HR_transform():
	return Compose ([ToTensor()])


def get_dataset(LR_image_dir=None, HR_image_dir=None, img_height=256, img_width=256, scale_factor=2):
	return DatasetFromFolder (LR_image_dir,
							  HR_image_dir,
							  LR_transform=LR_transform (img_height, img_width, scale_factor),
							  HR_transform=HR_transform ())
