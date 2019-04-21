import torch
import torch.nn as nn
import numpy as np

# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss (nn.Module):
	def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
		super (GANLoss, self).__init__ ()
		self.gan_type = gan_type.lower ()
		self.real_label_val = real_label_val
		self.fake_label_val = fake_label_val

		if self.gan_type == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss ()
		elif self.gan_type == 'lsgan':
			self.loss = nn.MSELoss ()
		elif self.gan_type == 'wgan-gp':

			def wgan_loss(input, target):
				# target is boolean
				return -1 * input.mean () if target else input.mean ()

			self.loss = wgan_loss
		else:
			raise NotImplementedError ('GAN type [{:s}] is not found'.format (self.gan_type))

	def get_target_label(self, input, target_is_real):
		if self.gan_type == 'wgan-gp':
			return target_is_real
		if target_is_real:
			return torch.empty_like (input).fill_ (self.real_label_val)
		else:
			return torch.empty_like (input).fill_ (self.fake_label_val)

	def forward(self, input, target_is_real):
		target_label = self.get_target_label (input, target_is_real)
		loss = self.loss (input, target_label)
		return loss


class GradientPenaltyLoss (nn.Module):
	def __init__(self, device=torch.device ('cpu')):
		super (GradientPenaltyLoss, self).__init__ ()
		self.register_buffer ('grad_outputs', torch.Tensor ())
		self.grad_outputs = self.grad_outputs.to (device)

	def get_grad_outputs(self, input):
		if self.grad_outputs.size () != input.size ():
			self.grad_outputs.resize_ (input.size ()).fill_ (1.0)
		return self.grad_outputs

	def forward(self, interp, interp_crit):
		grad_outputs = self.get_grad_outputs (interp_crit)
		grad_interp = torch.autograd.grad (outputs=interp_crit, inputs=interp,
										   grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
										   only_inputs=True)[0]
		grad_interp = grad_interp.view (grad_interp.size (0), -1)
		grad_interp_norm = grad_interp.norm (2, dim=1)

		loss = ((grad_interp_norm - 1) ** 2).mean ()
		return loss


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