"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

import _pickle as pickle

# pylint: disable=C0326
SEG_LABELS_LIST = [
    {"id": -1, "name": "void",       "rgb_values": [0,     0,    0]},
    {"id": 0 ,	"name":" bed                 ",	"rgb_values": [ 1 , 0, 0]},
	{"id": 1 ,	"name":" windowpane          ",	"rgb_values": [ 2 , 0, 0]},
	{"id": 2 ,	"name":" cabinet             ",	"rgb_values": [ 3 , 0, 0]},
	{"id": 3 ,	"name":" person              ",	"rgb_values": [ 4 , 0, 0]},
	{"id": 4 ,	"name":" door                ",	"rgb_values": [ 5 , 0, 0]},
	{"id": 5 ,	"name":" table               ",	"rgb_values": [ 6 , 0, 0]},
	{"id": 6 ,	"name":" curtain             ",	"rgb_values": [ 7 , 0, 0]},
	{"id": 7 ,	"name":" chair               ",	"rgb_values": [ 8 , 0, 0]},
	{"id": 8 ,	"name":" car                 ",	"rgb_values": [ 9 , 0, 0]},
	{"id": 9 ,	"name":" painting            ",	"rgb_values": [ 10 , 0, 0]},
	{"id": 10 ,	"name":" sofa                ",	"rgb_values": [ 11 , 0, 0]},
	{"id": 11 ,	"name":" shelf               ",	"rgb_values": [ 12 , 0, 0]},
	{"id": 12 ,	"name":" mirror              ",	"rgb_values": [ 13 , 0, 0]},
	{"id": 13 ,	"name":" armchair            ",	"rgb_values": [ 14 , 0, 0]},
	{"id": 14 ,	"name":" seat                ",	"rgb_values": [ 15 , 0, 0]},
	{"id": 15 ,	"name":" fence               ",	"rgb_values": [ 16 , 0, 0]},
	{"id": 16 ,	"name":" desk                ",	"rgb_values": [ 17 , 0, 0]},
	{"id": 17 ,	"name":" wardrobe            ",	"rgb_values": [ 18 , 0, 0]},
	{"id": 18 ,	"name":" lamp                ",	"rgb_values": [ 19 , 0, 0]},
	{"id": 19 ,	"name":" bathtub             ",	"rgb_values": [ 20 , 0, 0]},
	{"id": 20 ,	"name":" railing             ",	"rgb_values": [ 21 , 0, 0]},
	{"id": 21 ,	"name":" cushion             ",	"rgb_values": [ 22 , 0, 0]},
	{"id": 22 ,	"name":" box                 ",	"rgb_values": [ 23 , 0, 0]},
	{"id": 23 ,	"name":" column              ",	"rgb_values": [ 24 , 0, 0]},
	{"id": 24 ,	"name":" signboard           ",	"rgb_values": [ 25 , 0, 0]},
	{"id": 25 ,	"name":" chest of drawers    ",	"rgb_values": [ 26 , 0, 0]},
	{"id": 26 ,	"name":" counter             ",	"rgb_values": [ 27 , 0, 0]},
	{"id": 27 ,	"name":" sink                ",	"rgb_values": [ 28 , 0, 0]},
	{"id": 28 ,	"name":" fireplace           ",	"rgb_values": [ 29 , 0, 0]},
	{"id": 29 ,	"name":" refrigerator        ",	"rgb_values": [ 30 , 0, 0]},
	{"id": 30 ,	"name":" stairs              ",	"rgb_values": [ 31 , 0, 0]},
	{"id": 31 ,	"name":" case                ",	"rgb_values": [ 32 , 0, 0]},
	{"id": 32 ,	"name":" pool table          ",	"rgb_values": [ 33 , 0, 0]},
	{"id": 33 ,	"name":" pillow              ",	"rgb_values": [ 34 , 0, 0]},
	{"id": 34 ,	"name":" screen door         ",	"rgb_values": [ 35 , 0, 0]},
	{"id": 35 ,	"name":" bookcase            ",	"rgb_values": [ 36 , 0, 0]},
	{"id": 36 ,	"name":" coffee table        ",	"rgb_values": [ 37 , 0, 0]},
	{"id": 37 ,	"name":" toilet              ",	"rgb_values": [ 38 , 0, 0]},
	{"id": 38 ,	"name":" flower              ",	"rgb_values": [ 39 , 0, 0]},
	{"id": 39 ,	"name":" book                ",	"rgb_values": [ 40 , 0, 0]},
	{"id": 40 ,	"name":" bench               ",	"rgb_values": [ 41 , 0, 0]},
	{"id": 41 ,	"name":" countertop          ",	"rgb_values": [ 42 , 0, 0]},
	{"id": 42 ,	"name":" stove               ",	"rgb_values": [ 43 , 0, 0]},
	{"id": 43 ,	"name":" palm                ",	"rgb_values": [ 44 , 0, 0]},
	{"id": 44 ,	"name":" kitchen island      ",	"rgb_values": [ 45 , 0, 0]},
	{"id": 45 ,	"name":" computer            ",	"rgb_values": [ 46 , 0, 0]},
	{"id": 46 ,	"name":" swivel chair        ",	"rgb_values": [ 47 , 0, 0]},
	{"id": 47 ,	"name":" boat                ",	"rgb_values": [ 48 , 0, 0]},
	{"id": 48 ,	"name":" arcade machine      ",	"rgb_values": [ 49 , 0, 0]},
	{"id": 49 ,	"name":" bus                 ",	"rgb_values": [ 50 , 0, 0]},
	{"id": 50 ,	"name":" towel               ",	"rgb_values": [ 51 , 0, 0]},
	{"id": 51 ,	"name":" light               ",	"rgb_values": [ 52 , 0, 0]},
	{"id": 52 ,	"name":" truck               ",	"rgb_values": [ 53 , 0, 0]},
	{"id": 53 ,	"name":" chandelier          ",	"rgb_values": [ 54 , 0, 0]},
	{"id": 54 ,	"name":" awning              ",	"rgb_values": [ 55 , 0, 0]},
	{"id": 55 ,	"name":" streetlight         ",	"rgb_values": [ 56 , 0, 0]},
	{"id": 56 ,	"name":" booth               ",	"rgb_values": [ 57 , 0, 0]},
	{"id": 57 ,	"name":" television receiver ",	"rgb_values": [ 58 , 0, 0]},
	{"id": 58 ,	"name":" airplane            ",	"rgb_values": [ 59 , 0, 0]},
	{"id": 59 ,	"name":" apparel             ",	"rgb_values": [ 60 , 0, 0]},
	{"id": 60 ,	"name":" pole                ",	"rgb_values": [ 61 , 0, 0]},
	{"id": 61 ,	"name":" bannister           ",	"rgb_values": [ 62 , 0, 0]},
	{"id": 62 ,	"name":" ottoman             ",	"rgb_values": [ 63 , 0, 0]},
	{"id": 63 ,	"name":" bottle              ",	"rgb_values": [ 64 , 0, 0]},
	{"id": 64 ,	"name":" van                 ",	"rgb_values": [ 65 , 0, 0]},
	{"id": 65 ,	"name":" ship                ",	"rgb_values": [ 66 , 0, 0]},
	{"id": 66 ,	"name":" fountain            ",	"rgb_values": [ 67 , 0, 0]},
	{"id": 67 ,	"name":" washer              ",	"rgb_values": [ 68 , 0, 0]},
	{"id": 68 ,	"name":" plaything           ",	"rgb_values": [ 69 , 0, 0]},
	{"id": 69 ,	"name":" stool               ",	"rgb_values": [ 70 , 0, 0]},
	{"id": 70 ,	"name":" barrel              ",	"rgb_values": [ 71 , 0, 0]},
	{"id": 71 ,	"name":" basket              ",	"rgb_values": [ 72 , 0, 0]},
	{"id": 72 ,	"name":" bag                 ",	"rgb_values": [ 73 , 0, 0]},
	{"id": 73 ,	"name":" minibike            ",	"rgb_values": [ 74 , 0, 0]},
	{"id": 74 ,	"name":" oven                ",	"rgb_values": [ 75 , 0, 0]},
	{"id": 75 ,	"name":" ball                ",	"rgb_values": [ 76 , 0, 0]},
	{"id": 76 ,	"name":" food                ",	"rgb_values": [ 77 , 0, 0]},
	{"id": 77 ,	"name":" step                ",	"rgb_values": [ 78 , 0, 0]},
	{"id": 78 ,	"name":" trade name          ",	"rgb_values": [ 79 , 0, 0]},
	{"id": 79 ,	"name":" microwave           ",	"rgb_values": [ 80 , 0, 0]},
	{"id": 80 ,	"name":" pot                 ",	"rgb_values": [ 81 , 0, 0]},
	{"id": 81 ,	"name":" animal              ",	"rgb_values": [ 82 , 0, 0]},
	{"id": 82 ,	"name":" bicycle             ",	"rgb_values": [ 83 , 0, 0]},
	{"id": 83 ,	"name":" dishwasher          ",	"rgb_values": [ 84 , 0, 0]},
	{"id": 84 ,	"name":" screen              ",	"rgb_values": [ 85 , 0, 0]},
	{"id": 85 ,	"name":" sculpture           ",	"rgb_values": [ 86 , 0, 0]},
	{"id": 86 ,	"name":" hood                ",	"rgb_values": [ 87 , 0, 0]},
	{"id": 87 ,	"name":" sconce              ",	"rgb_values": [ 88 , 0, 0]},
	{"id": 88 ,	"name":" vase                ",	"rgb_values": [ 89 , 0, 0]},
	{"id": 89 ,	"name":" traffic light       ",	"rgb_values": [ 90 , 0, 0]},
	{"id": 90 ,	"name":" tray                ",	"rgb_values": [ 91 , 0, 0]},
	{"id": 91 ,	"name":" ashcan              ",	"rgb_values": [ 92 , 0, 0]},
	{"id": 92 ,	"name":" fan                 ",	"rgb_values": [ 93 , 0, 0]},
	{"id": 93 ,	"name":" plate               ",	"rgb_values": [ 94 , 0, 0]},
	{"id": 94 ,	"name":" monitor             ",	"rgb_values": [ 95 , 0, 0]},
	{"id": 95 ,	"name":" bulletin board      ",	"rgb_values": [ 96 , 0, 0]},
	{"id": 96 ,	"name":" radiator            ",	"rgb_values": [ 97 , 0, 0]},
	{"id": 97 ,	"name":" glass               ",	"rgb_values": [ 98 , 0, 0]},
	{"id": 98 ,	"name":" clock               ",	"rgb_values": [ 99 , 0, 0]},
	{"id": 99 ,	"name":" flag                ",	"rgb_values": [ 100 , 0, 0]}]

SEG_LABELS_LIST2 = [
    {"id": -1, "name": "void	",       			"rgb_values": [0,     0,    0]},
    {"id": 0 ,	"name":" bed                 ",	"rgb_values": [ 1 ,  86 ,  235 ]},
	{"id": 1 ,	"name":" windowpane          ",	"rgb_values": [ 2 ,  164 ,  106 ]},
	{"id": 2 ,	"name":" cabinet             ",	"rgb_values": [ 3 ,  67 ,  162 ]},
	{"id": 3 ,	"name":" person              ",	"rgb_values": [ 4 ,  29 ,  220 ]},
	{"id": 4 ,	"name":" door                ",	"rgb_values": [ 5 ,  116 ,  224 ]},
	{"id": 5 ,	"name":" table               ",	"rgb_values": [ 6 ,  136 ,  33 ]},
	{"id": 6 ,	"name":" curtain             ",	"rgb_values": [ 7 ,  102 ,  183 ]},
	{"id": 7 ,	"name":" chair               ",	"rgb_values": [ 8 ,  156 ,  114 ]},
	{"id": 8 ,	"name":" car                 ",	"rgb_values": [ 9 ,  195 ,  146 ]},
	{"id": 9 ,	"name":" painting            ",	"rgb_values": [ 10 ,  11 ,  162 ]},
	{"id": 10 ,	"name":" sofa                ",	"rgb_values": [ 11 ,  135 ,  73 ]},
	{"id": 11 ,	"name":" shelf               ",	"rgb_values": [ 12 ,  39 ,  213 ]},
	{"id": 12 ,	"name":" mirror              ",	"rgb_values": [ 13 ,  68 ,  175 ]},
	{"id": 13 ,	"name":" armchair            ",	"rgb_values": [ 14 ,  126 ,  130 ]},
	{"id": 14 ,	"name":" seat                ",	"rgb_values": [ 15 ,  253 ,  188 ]},
	{"id": 15 ,	"name":" fence               ",	"rgb_values": [ 16 ,  91 ,  165 ]},
	{"id": 16 ,	"name":" desk                ",	"rgb_values": [ 17 ,  144 ,  175 ]},
	{"id": 17 ,	"name":" wardrobe            ",	"rgb_values": [ 18 ,  40 ,  254 ]},
	{"id": 18 ,	"name":" lamp                ",	"rgb_values": [ 19 ,  8 ,  186 ]},
	{"id": 19 ,	"name":" bathtub             ",	"rgb_values": [ 20 ,  119 ,  71 ]},
	{"id": 20 ,	"name":" railing             ",	"rgb_values": [ 21 ,  219 ,  75 ]},
	{"id": 21 ,	"name":" cushion             ",	"rgb_values": [ 22 ,  94 ,  218 ]},
	{"id": 22 ,	"name":" box                 ",	"rgb_values": [ 23 ,  83 ,  26 ]},
	{"id": 23 ,	"name":" column              ",	"rgb_values": [ 24 ,  21 ,  132 ]},
	{"id": 24 ,	"name":" signboard           ",	"rgb_values": [ 25 ,  70 ,  100 ]},
	{"id": 25 ,	"name":" chest of drawers    ",	"rgb_values": [ 26 ,  59 ,  62 ]},
	{"id": 26 ,	"name":" counter             ",	"rgb_values": [ 27 ,  113 ,  211 ]},
	{"id": 27 ,	"name":" sink                ",	"rgb_values": [ 28 ,  56 ,  243 ]},
	{"id": 28 ,	"name":" fireplace           ",	"rgb_values": [ 29 ,  72 ,  236 ]},
	{"id": 29 ,	"name":" refrigerator        ",	"rgb_values": [ 30 ,  224 ,  33 ]},
	{"id": 30 ,	"name":" stairs              ",	"rgb_values": [ 31 ,  39 ,  215 ]},
	{"id": 31 ,	"name":" case                ",	"rgb_values": [ 32 ,  229 ,  31 ]},
	{"id": 32 ,	"name":" pool table          ",	"rgb_values": [ 33 ,  57 ,  18 ]},
	{"id": 33 ,	"name":" pillow              ",	"rgb_values": [ 34 ,  139 ,  81 ]},
	{"id": 34 ,	"name":" screen door         ",	"rgb_values": [ 35 ,  9 ,  107 ]},
	{"id": 35 ,	"name":" bookcase            ",	"rgb_values": [ 36 ,  197 ,  228 ]},
	{"id": 36 ,	"name":" coffee table        ",	"rgb_values": [ 37 ,  71 ,  53 ]},
	{"id": 37 ,	"name":" toilet              ",	"rgb_values": [ 38 ,  226 ,  191 ]},
	{"id": 38 ,	"name":" flower              ",	"rgb_values": [ 39 ,  194 ,  212 ]},
	{"id": 39 ,	"name":" book                ",	"rgb_values": [ 40 ,  241 ,  117 ]},
	{"id": 40 ,	"name":" bench               ",	"rgb_values": [ 41 ,  165 ,  193 ]},
	{"id": 41 ,	"name":" countertop          ",	"rgb_values": [ 42 ,  65 ,  197 ]},
	{"id": 42 ,	"name":" stove               ",	"rgb_values": [ 43 ,  185 ,  206 ]},
	{"id": 43 ,	"name":" palm                ",	"rgb_values": [ 44 ,  251 ,  167 ]},
	{"id": 44 ,	"name":" kitchen island      ",	"rgb_values": [ 45 ,  187 ,  19 ]},
	{"id": 45 ,	"name":" computer            ",	"rgb_values": [ 46 ,  9 ,  180 ]},
	{"id": 46 ,	"name":" swivel chair        ",	"rgb_values": [ 47 ,  133 ,  250 ]},
	{"id": 47 ,	"name":" boat                ",	"rgb_values": [ 48 ,  103 ,  89 ]},
	{"id": 48 ,	"name":" arcade machine      ",	"rgb_values": [ 49 ,  132 ,  38 ]},
	{"id": 49 ,	"name":" bus                 ",	"rgb_values": [ 50 ,  88 ,  225 ]},
	{"id": 50 ,	"name":" towel               ",	"rgb_values": [ 51 ,  81 ,  184 ]},
	{"id": 51 ,	"name":" light               ",	"rgb_values": [ 52 ,  95 ,  61 ]},
	{"id": 52 ,	"name":" truck               ",	"rgb_values": [ 53 ,  6 ,  124 ]},
	{"id": 53 ,	"name":" chandelier          ",	"rgb_values": [ 54 ,  97 ,  20 ]},
	{"id": 54 ,	"name":" awning              ",	"rgb_values": [ 55 ,  172 ,  50 ]},
	{"id": 55 ,	"name":" streetlight         ",	"rgb_values": [ 56 ,  110 ,  78 ]},
	{"id": 56 ,	"name":" booth               ",	"rgb_values": [ 57 ,  240 ,  113 ]},
	{"id": 57 ,	"name":" television receiver ",	"rgb_values": [ 58 ,  36 ,  251 ]},
	{"id": 58 ,	"name":" airplane            ",	"rgb_values": [ 59 ,  129 ,  84 ]},
	{"id": 59 ,	"name":" apparel             ",	"rgb_values": [ 60 ,  245 ,  212 ]},
	{"id": 60 ,	"name":" pole                ",	"rgb_values": [ 61 ,  29 ,  251 ]},
	{"id": 61 ,	"name":" bannister           ",	"rgb_values": [ 62 ,  182 ,  210 ]},
	{"id": 62 ,	"name":" ottoman             ",	"rgb_values": [ 63 ,  250 ,  121 ]},
	{"id": 63 ,	"name":" bottle              ",	"rgb_values": [ 64 ,  162 ,  209 ]},
	{"id": 64 ,	"name":" van                 ",	"rgb_values": [ 65 ,  44 ,  43 ]},
	{"id": 65 ,	"name":" ship                ",	"rgb_values": [ 66 ,  93 ,  10 ]},
	{"id": 66 ,	"name":" fountain            ",	"rgb_values": [ 67 ,  24 ,  148 ]},
	{"id": 67 ,	"name":" washer              ",	"rgb_values": [ 68 ,  128 ,  254 ]},
	{"id": 68 ,	"name":" plaything           ",	"rgb_values": [ 69 ,  124 ,  219 ]},
	{"id": 69 ,	"name":" stool               ",	"rgb_values": [ 70 ,  175 ,  37 ]},
	{"id": 70 ,	"name":" barrel              ",	"rgb_values": [ 71 ,  78 ,  90 ]},
	{"id": 71 ,	"name":" basket              ",	"rgb_values": [ 72 ,  120 ,  56 ]},
	{"id": 72 ,	"name":" bag                 ",	"rgb_values": [ 73 ,  150 ,  240 ]},
	{"id": 73 ,	"name":" minibike            ",	"rgb_values": [ 74 ,  142 ,  97 ]},
	{"id": 74 ,	"name":" oven                ",	"rgb_values": [ 75 ,  201 ,  3 ]},
	{"id": 75 ,	"name":" ball                ",	"rgb_values": [ 76 ,  144 ,  133 ]},
	{"id": 76 ,	"name":" food                ",	"rgb_values": [ 77 ,  147 ,  6 ]},
	{"id": 77 ,	"name":" step                ",	"rgb_values": [ 78 ,  19 ,  228 ]},
	{"id": 78 ,	"name":" trade name          ",	"rgb_values": [ 79 ,  169 ,  183 ]},
	{"id": 79 ,	"name":" microwave           ",	"rgb_values": [ 80 ,  196 ,  215 ]},
	{"id": 80 ,	"name":" pot                 ",	"rgb_values": [ 81 ,  205 ,  63 ]},
	{"id": 81 ,	"name":" animal              ",	"rgb_values": [ 82 ,  9 ,  78 ]},
	{"id": 82 ,	"name":" bicycle             ",	"rgb_values": [ 83 ,  137 ,  31 ]},
	{"id": 83 ,	"name":" dishwasher          ",	"rgb_values": [ 84 ,  137 ,  202 ]},
	{"id": 84 ,	"name":" screen              ",	"rgb_values": [ 85 ,  90 ,  215 ]},
	{"id": 85 ,	"name":" sculpture           ",	"rgb_values": [ 86 ,  84 ,  201 ]},
	{"id": 86 ,	"name":" hood                ",	"rgb_values": [ 87 ,  100 ,  20 ]},
	{"id": 87 ,	"name":" sconce              ",	"rgb_values": [ 88 ,  13 ,  207 ]},
	{"id": 88 ,	"name":" vase                ",	"rgb_values": [ 89 ,  203 ,  185 ]},
	{"id": 89 ,	"name":" traffic light       ",	"rgb_values": [ 90 ,  233 ,  170 ]},
	{"id": 90 ,	"name":" tray                ",	"rgb_values": [ 91 ,  110 ,  33 ]},
	{"id": 91 ,	"name":" ashcan              ",	"rgb_values": [ 92 ,  35 ,  245 ]},
	{"id": 92 ,	"name":" fan                 ",	"rgb_values": [ 93 ,  13 ,  183 ]},
	{"id": 93 ,	"name":" plate               ",	"rgb_values": [ 94 ,  203 ,  241 ]},
	{"id": 94 ,	"name":" monitor             ",	"rgb_values": [ 95 ,  148 ,  62 ]},
	{"id": 95 ,	"name":" bulletin board      ",	"rgb_values": [ 96 ,  14 ,  222 ]},
	{"id": 96 ,	"name":" radiator            ",	"rgb_values": [ 97 ,  168 ,  95 ]},
	{"id": 97 ,	"name":" glass               ",	"rgb_values": [ 98 ,  246 ,  19 ]},
	{"id": 98 ,	"name":" clock               ",	"rgb_values": [ 99 ,  212 ,  92 ]},
	{"id": 99 ,	"name":" flag                ",	"rgb_values": [ 100 ,  48 ,  119 ]}]


def train_label_img_to_rgb(label_img):
    label_img = np.squeeze(label_img)
    labels = np.unique(label_img)
    label_infos = [l for l in SEG_LABELS_LIST2 if l['id'] in labels]

    label_img_rgb = np.array([label_img,
                              label_img,
                              label_img]).transpose(1,2,0)
    for l in label_infos:
        mask = label_img == l['id']
        label_img_rgb[mask] = l['rgb_values']

    return label_img_rgb.astype(np.uint8)


class SegmentationtrainData(data.Dataset):

	def __init__(self, image_paths_file):
		self.root_dir_name = os.path.dirname(image_paths_file)

		with open(image_paths_file) as f:
			self.image_names = f.read().splitlines()

	def __getitem__(self, key):
		if isinstance(key, slice):
            # get the start, stop, and step from the slice
			return [self[ii] for ii in range(*key.indices(len(self)))]
		elif isinstance(key, int):
            # handle negative indices
			if key < 0:
				key += len(self)
			if key < 0 or key >= len(self):
				raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
			return self.get_item_from_index(key)
		else:
			raise TypeError("Invalid argument type.")

	def __len__(self):
		return len(self.image_names)

	def get_item_from_index(self, index):
		to_tensor = transforms.ToTensor()
		img_id = self.image_names[index].replace('.jpg', '')

		img = Image.open(os.path.join(self.root_dir_name,
									  'images',
                                      img_id + '.jpg'))
		
		if img.mode != 'RGB':
			img = img.convert('RGB')
		#center_crop = transforms.CenterCrop(256)
		#img = center_crop(img)
		resize = transforms.Resize((256,256))
		img = resize(img)
		img = to_tensor(img)

		target = Image.open(os.path.join(self.root_dir_name,
										 'annotations_instance',
                                         img_id + '.png'))
		if target.mode != 'RGB':
			target = target.convert('RGB')
		r, g, b =target.split()
        #print(b)
		#Image.new(im1.mode, im1.size, "#000000")  # 黑色
		e = Image.new('L',target.size,"BLACK")
        #e = Image.new("RGB",,(0,0,0))
		target = Image.merge("RGB",(r,e,e))
		resize = transforms.Resize((256,256),transforms.InterpolationMode.NEAREST)
		target = resize(target)
		#target = center_crop(target)
		target = np.array(target, dtype=np.int64)
        
		target_labels = target[..., 0]

		for label in SEG_LABELS_LIST:
			mask = np.all(target == label['rgb_values'], axis=2)
			target_labels[mask] = label['id']
		#target_labels[target_labels == -1] = 255

		target_labels = torch.from_numpy(target_labels.copy())

		return img, target_labels
