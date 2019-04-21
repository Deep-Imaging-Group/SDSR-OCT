import torch
import torch.nn as nn
import numpy as np
import visdom
import math
from feature_extractor_vgg import vgg19

class FeatureExtractor (nn.Module):
	def __init__(self):
		super (FeatureExtractor, self).__init__ ()

		vgg19_model = vgg19 (pretrained=False)

		# Extracts features at the 11th layer
		self.feature_extractor = nn.Sequential (*list (vgg19_model.features.children ())[:12])

	def forward(self, img):
		out = self.feature_extractor (img)
		return out


class Residual_Block (nn.Module):
	def __init__(self, in_nc, mid_nc, kernel_size=3, stride=1, padding=1, neg_slope=0.2, res_scale=0.2):
		super (Residual_Block, self).__init__ ()
		self.res_scale = res_scale
		# block1
		self.conv_block1 = nn.Sequential (
			nn.Conv2d (in_channels=in_nc, out_channels=mid_nc, kernel_size=kernel_size, stride=stride, padding=padding),
			nn.LeakyReLU (negative_slope=neg_slope, inplace=False))
		# block2
		self.conv_block2 = nn.Sequential (
			nn.Conv2d (in_channels=in_nc + mid_nc * 1, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
					   padding=padding),
			nn.LeakyReLU (negative_slope=neg_slope, inplace=False))
		# block3
		self.conv_block3 = nn.Sequential (
			nn.Conv2d (in_channels=in_nc + mid_nc * 2, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
					   padding=padding),
			nn.LeakyReLU (negative_slope=neg_slope, inplace=False))
		# block4
		self.conv_block4 = nn.Sequential (
			nn.Conv2d (in_channels=in_nc + mid_nc * 3, out_channels=mid_nc, kernel_size=kernel_size, stride=stride,
					   padding=padding),
			nn.LeakyReLU (negative_slope=neg_slope, inplace=False))
		# block5
		self.conv_block5 = nn.Sequential (
			nn.Conv2d (in_channels=in_nc + mid_nc * 4, out_channels=in_nc, kernel_size=kernel_size, stride=stride,
					   padding=padding))

	def forward(self, x):
		x1 = self.conv_block1 (x)
		x2 = self.conv_block2 (torch.cat ((x, x1), 1))
		x3 = self.conv_block3 (torch.cat ((x, x1, x2), 1))
		x4 = self.conv_block4 (torch.cat ((x, x1, x2, x3), 1))
		x5 = self.conv_block5 (torch.cat ((x, x1, x2, x3, x4), 1))
		out = x5.mul (self.res_scale) + x
		return out


class RRDB_Block (nn.Module):
	def __init__(self, in_nc, mid_nc, kernel_size=3, stride=1, padding=1, neg_slope=0.2, res_scale=0.2,
				 num_residuals=3):
		super (RRDB_Block, self).__init__ ()
		self.res_scale = res_scale
		residual_layers = []
		for num in range (num_residuals):
			residual_layers.append (
				Residual_Block (in_nc, mid_nc, kernel_size, stride, padding, neg_slope, self.res_scale))
		self.Residual_Blocks = nn.Sequential (*residual_layers)

	def forward(self, x):
		out = self.Residual_Blocks (x)
		return out.mul (self.res_scale) + x


class UpUint (nn.Module):
	def __init__(self, num_input_features, num_output_features, kernel_size, stride, padding):
		super (UpUint, self).__init__ ()
		self.deconv1 = nn.ConvTranspose2d (in_channels=num_input_features, out_channels=num_output_features,
										   kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu1 = nn.PReLU ()
		self.conv1 = nn.Conv2d (in_channels=num_output_features, out_channels=num_input_features,
								kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu2 = nn.PReLU ()
		self.deconv2 = nn.ConvTranspose2d (in_channels=num_input_features, out_channels=num_output_features,
										   kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu3 = nn.PReLU ()

	def forward(self, x):
		h0 = self.relu1 (self.deconv1 (x))
		l0 = self.relu2 (self.conv1 (h0))
		diff = l0 - x
		h1 = self.relu3 (self.deconv2 (diff))
		out = h1 + h0
		return out


class DownUint (nn.Module):
	def __init__(self, num_input_features, num_output_features, kernel_size, stride, padding):
		super (DownUint, self).__init__ ()
		self.conv1 = nn.Conv2d (in_channels=num_input_features, out_channels=num_output_features,
								kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu1 = nn.PReLU ()
		self.deconv1 = nn.ConvTranspose2d (in_channels=num_output_features, out_channels=num_input_features,
										   kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu2 = nn.PReLU ()
		self.conv2 = nn.Conv2d (in_channels=num_input_features, out_channels=num_output_features,
								kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
		self.relu3 = nn.PReLU ()

	def forward(self, x):
		l0 = self.relu1 (self.conv1 (x))
		h0 = self.relu2 (self.deconv1 (l0))
		diff = h0 - x
		l1 = self.relu3 (self.conv2 (diff))
		out = l1 + l0
		return out


class TranSition (nn.Module):
	def __init__(self, num_input_features, num_output_features):
		super (TranSition, self).__init__ ()
		self.conv1 = nn.Conv2d (in_channels=num_input_features, out_channels=num_output_features, kernel_size=1,
								stride=1, padding=0, bias=False)
		self.relu1 = nn.PReLU ()

	def forward(self, x):
		out = self.relu1 (self.conv1 (x))
		return out


class Generator (nn.Module):
	def __init__(self, in_nc=1, mid_nc=64, out_nc=1, scale_factor=2, num_RRDBS=10):
		super (Generator, self).__init__ ()

		if scale_factor == 2:
			self.kernel_size = 6
			self.stride = 2
			self.padding = 2
		elif scale_factor == 4:
			self.kernel_size = 8
			self.stride = 4
			self.padding = 2
		elif scale_factor == 8:
			self.kernel_size = 12
			self.stride = 8
			self.padding = 2

		RRDB_layers = []
		self.block1 = nn.Conv2d (in_channels=in_nc, out_channels=mid_nc, kernel_size=3, stride=1, padding=1)
		for num in range (num_RRDBS):
			RRDB_layers.append (RRDB_Block (in_nc=mid_nc, mid_nc=mid_nc))
		self.block2 = nn.Sequential (*RRDB_layers)

		self.trans11 = TranSition (64, 64)
		self.up1 = UpUint (64, 64, self.kernel_size, self.stride, self.padding)
		self.trans12 = TranSition (64, 64)
		self.down1 = DownUint (64, 64, self.kernel_size, self.stride, self.padding)

		self.trans21 = TranSition (128, 64)
		self.up2 = UpUint (64, 64, self.kernel_size, self.stride, self.padding)
		self.trans22 = TranSition (128, 64)
		self.down2 = DownUint (64, 64, self.kernel_size, self.stride, self.padding)
		self.trans3 = TranSition (192, 64)

		self.up3 = UpUint (64, 64, self.kernel_size, self.stride, self.padding)
		self.trans4 = TranSition (192, 64)
		self.reconv = nn.Conv2d (64, out_nc, kernel_size=3, stride=1, padding=1, bias=False)



	def forward(self, x):
		x = self.block1 (x)
		features = self.block2 (x)

		trans11 = self.trans11 (features)
		h1 = self.up1 (trans11)
		trans12 = self.trans12 (h1)
		l1 = self.down1 (trans12)
		cat_l = torch.cat ((features, l1), 1)

		trans21 = self.trans21 (cat_l)
		h2 = self.up2 (trans21)
		cat_h = torch.cat ((h1, h2), 1)
		trans22 = self.trans22 (cat_h)
		l2 = self.down2 (trans22)
		cat_l = torch.cat ((cat_l, l2), 1)

		trans3 = self.trans3 (cat_l)
		h3 = self.up3 (trans3)
		cat_h = torch.cat ((cat_h, h3), 1)

		hout = self.trans4 (cat_h)
		out = self.reconv (hout)

		return out


class Discriminator (nn.Module):
	def __init__(self, in_channels=1):
		super (Discriminator, self).__init__ ()

		def discriminator_block(in_filters, out_filters, stride, normalize):
			"""Returns layers of each discriminator block"""
			layers = [nn.Conv2d (in_filters, out_filters, 3, stride, 1)]
			if normalize:
				layers.append (nn.BatchNorm2d (out_filters))
			layers.append (nn.LeakyReLU (0.2, inplace=True))
			return layers

		layers = []
		in_filters = in_channels
		for out_filters, stride, normalize in [(64, 1, False),
											   (64, 2, True),
											   (128, 1, True),
											   (128, 2, True),
											   (256, 1, True),
											   (256, 2, True),
											   (512, 1, True),
											   (512, 2, True), ]:
			layers.extend (discriminator_block (in_filters, out_filters, stride, normalize))
			in_filters = out_filters

		# Output layer
		layers.append (nn.Conv2d (out_filters, 1, 3, 1, 1))
		# layers.append (nn.Sigmoid ())

		self.model = nn.Sequential (*layers)

	def forward(self, img):
		return self.model (img)


class L1_Charbonnier_loss (nn.Module):
	"""L1 Charbonnierloss."""

	def __init__(self):
		super (L1_Charbonnier_loss, self).__init__ ()
		self.eps = 1e-6

	def forward(self, X, Y):
		diff = torch.add (X, -Y)
		error = torch.sqrt (diff * diff + self.eps)
		loss = torch.mean (error)
		return loss


class TV_loss (nn.Module):
	def __init__(self, tv_loss_weight=1):
		super (TV_loss, self).__init__ ()
		self.tv_loss_weight = tv_loss_weight

	def forward(self, x):
		batch_size = x.size ()[0]
		h_x = x.size ()[2]
		w_x = x.size ()[3]
		count_h = self.tensor_size (x[:, :, 1:, :])
		count_w = self.tensor_size (x[:, :, :, 1:])
		h_tv = torch.pow ((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum ()
		w_tv = torch.pow ((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum ()
		return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class edgeV_loss (nn.Module):
	def __init__(self):
		super (edgeV_loss, self).__init__ ()

	def forward(self, X1, X2):
		X1_up = X1[:, :, :-1, :]
		X1_down = X1[:, :, 1:, :]
		X2_up = X2[:, :, :-1, :]
		X2_down = X2[:, :, 1:, :]
		return -np.log (int (torch.sum (torch.abs (X1_up - X1_down))) / int (torch.sum (torch.abs (X2_up - X2_down))))


class Visualizer (object):
	def __init__(self, env='default', **kwargs):
		self.vis = visdom.Visdom (env=env, **kwargs)
		self.index = {}

	def plot_many_stack(self, d):
		''' self.plot('loss',1.00) '''
		name = list (d.keys ())
		name_total = " ".join (name)
		x = self.index.get (name_total, 0)
		val = list (d.values ())
		if len (val) == 1:
			y = np.array (val)
		else:
			y = np.array (val).reshape (-1, len (val))
		# print(x)
		self.vis.line (Y=y, X=np.ones (y.shape) * x, win=str (name_total), opts=dict (legend=name, title=name_total),
					   update=None if x == 0 else 'append')
		self.index[name_total] = x + 1


if __name__ == '__main__':
	model = Generator ()
	print (model)
