import argparse
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import os
import cv2
import numpy as np
import random
from torchvision import transforms
from skimage import io as sio
from models import Generator, Discriminator, FeatureExtractor, L1_Charbonnier_loss, TV_loss, edgeV_loss, Visualizer
from dataset import get_dataset
from vis_tools import Visualizer
from loss import GANLoss, GradientPenaltyLoss

parser = argparse.ArgumentParser (description='pytorch SDSR-OCT')
parser.add_argument('--gan_type', type=str, default='vanilla')
parser.add_argument('--GANLoss_weights', type=float, default=5e-3)
parser.add_argument('--pixel_loss_type', type=str, default='L2')
parser.add_argument('--pixel_loss_weights', type=float, default=1e-2)
parser.add_argument('--content_loss_type', type=str, default='L1_Charbonnier')
parser.add_argument('--content_loss_weights', type=float, default=1)
parser.add_argument('--in_nc', type=int, default=1, help='number of input image channels')
parser.add_argument('--mid_nc', type=int, default=64, help='number of middle feature maps')
parser.add_argument('--out_nc', type=int, default=1, help='number of output image channels')
parser.add_argument('--num_RRDBs', type=int, default=10, help='number of RRDB layers')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument ('--num_epochs', type=int, default=500, help='number of training epochs')
parser.add_argument ('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument ('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument ('--resume_gen', type=str, default='', help='path to generator checkpoint')
parser.add_argument ('--resume_dis', type=str, default='', help='path to discriminator checkpoint')
parser.add_argument ('--start_epoch', type=int, default=1, help='restart epoch number for training')
parser.add_argument ('--threads', type=int, default=0, help='number of threads')
parser.add_argument ('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument ('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument ('--step', type=int, default=50, help='Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10')
parser.add_argument ('--pretrained_gen', type=str, default='', help='path to generator parameters')
parser.add_argument ('--pretrained_dis', type=str, default='', help='path to discriminator parameters')
parser.add_argument('--img_height', type=int, default=256, help='height of HR images')
parser.add_argument('--img_width', type=int, default=256, help='width of HR images')
parser.add_argument ('--scale_factor', type=int, default=8, help='scale factor')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--n_critic', type=int, default=1, help='number of training steps for discriminator per iter')
parser.add_argument ('--LR_train_dir', type=str, default='../dataset/train/', help='LR image path to training data directory')
parser.add_argument ('--HR_train_dir', type=str, default='../dataset/train_label/', help='HR image path to training data directory')
parser.add_argument ('--LR_test_dir', type=str, default='../dataset/test/', help='LR image path to testing data directory')
parser.add_argument ('--HR_test_dir', type=str, default='../dataset/test_label/', help='HR image path to testing data directory')
parser.add_argument ('--train_interval', type=int, default=50, help='interval for training to save image')
parser.add_argument ('--test_interval', type=int, default=10, help='interval for testing to save image')
opt = parser.parse_args ()

# print(opt)

# ...

seed = random.randint (1, 10000)
print ("Random Seed: ", seed)
torch.manual_seed (seed)
if opt.cuda:
	torch.cuda.manual_seed (seed)


# build network
print('==>building network...')
generator = Generator(in_nc=opt.in_nc, mid_nc=opt.mid_nc, out_nc=opt.out_nc, scale_factor=opt.scale_factor, num_RRDBS=opt.num_RRDBs)
discriminator = Discriminator ()
feature_extractor = FeatureExtractor()


# loss

# content loss
if opt.content_loss_type == 'L1_Charbonnier':
	content_loss = L1_Charbonnier_loss ()
elif opt.content_loss_type == 'L1':
	content_loss = torch.nn.L1Loss ()
elif opt.content_loss_type == 'L2':
	content_loss = torch.nn.MSELoss ()

# pixel loss
if opt.pixel_loss_type == 'L1':
	pixel_loss = torch.nn.L1Loss ()
elif opt.pixel_loss_type == 'L2':
	pixel_loss = torch.nn.MSELoss ()

# gan loss
GAN_loss = GANLoss(opt.gan_type, real_label_val=1.0, fake_label_val=0.0)
edge_loss = edgeV_loss()
tv_loss = TV_loss()
# GPU
if opt.cuda and not torch.cuda.is_available():  # 检查是否有GPU
	raise Exception('No GPU found, please run without --cuda')
print("===> Setting GPU")
if opt.cuda:
	print('cuda_mode:', opt.cuda)
	generator = generator.cuda()
	discriminator = discriminator.cuda()
	feature_extractor = feature_extractor.cuda()
	content_loss = content_loss.cuda()
	pixel_loss = pixel_loss.cuda()
	GAN_loss = GAN_loss.cuda()
	edge_loss = edge_loss.cuda()
	tv_loss = tv_loss.cuda()

# optimizer
print("===> Setting Optimizer")
Gen_optim = torch.optim.Adam(generator.parameters(), lr=opt.lr)
Dis_optim = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

# visualizer
train_vis = Visualizer (env='training')


# training
def train(train_dataloader, generator, discriminator, Gen_optim, Dis_optim, content_loss, pixel_loss, save_img_dir):
	print('==>Training...')
	for epoch in range(opt.start_epoch, opt.num_epochs + 1):
		train_process(train_dataloader, generator, discriminator, Gen_optim, Dis_optim, content_loss, pixel_loss, save_img_dir, epoch, epochs=opt.num_epochs, interval=opt.train_interval)
		save_checkpoint(generator, discriminator, epoch)


# testing
def test(test_dataloader, generator, save_img_dir):
	print ('==>Testing...')
	test_process (test_dataloader, generator, save_img_dir)


# train every epoch
def train_process(dataloader, generator, discriminator, Gen_optim, Dis_optim, content_loss, pixel_loss, save_img_dir, epoch=1, epochs=1, interval=100):
	lr = adjust_learning_rate (epoch - 1)
	for param_group in Gen_optim.param_groups:
		param_group["lr"] = lr
		print ("epoch =", epoch, "lr =", Gen_optim.param_groups[0]["lr"])
	for iteration, (inputs, labels) in enumerate (dataloader):
		inputs = Variable (inputs)  # input
		real_imgs = Variable (labels)  # label

		if opt.cuda:
			inputs = inputs.cuda ()
			real_imgs = real_imgs.cuda ()
		# -----------------------------------------------
		# training discriminator
		# ------------------------------------------------
		fake_imgs = generator (inputs)  # network output

		Dis_optim.zero_grad ()

		real_out = discriminator(real_imgs)
		fake_out = discriminator (fake_imgs)

		real_loss = GAN_loss (real_out - torch.mean (fake_out), True)
		fake_loss = GAN_loss (fake_out - torch.mean (real_out), False)

		d_loss = (real_loss + fake_loss) / 2

		d_loss.backward(retain_graph=True)
		Dis_optim.step()

		# Clip weights of discriminator
		for p in discriminator.parameters ():
			p.data.clamp_ (-opt.clip_value, opt.clip_value)

		if iteration % opt.n_critic == 0:

			# ----------------------------------------------------------------------
			# training generator
			# ----------------------------------------------------------------------
			Gen_optim.zero_grad ()
			fake_imgs = generator(inputs)

			loss_pixel = pixel_loss(real_imgs, fake_imgs)

			gen_features = feature_extractor (fake_imgs)
			real_features = Variable (feature_extractor (real_imgs).data, requires_grad=False)
			loss_content = content_loss (gen_features, real_features)

			real_out = discriminator (real_imgs)
			fake_out = discriminator (fake_imgs)

			real_loss = GAN_loss (real_out - torch.mean (fake_out), True)
			fake_loss = GAN_loss (fake_out - torch.mean (real_out), False)

			g_loss = opt.pixel_loss_weights*loss_pixel + opt.content_loss_weights*loss_content + opt.GANLoss_weights * (real_loss + fake_loss) / 2
			g_loss.backward ()
			Gen_optim.step ()

			psnr = PSNR(fake_imgs, real_imgs)

			train_vis.plot ('discriminator loss', d_loss.item ())
			train_vis.plot ('generator loss', g_loss.item ())
			train_vis.plot ('psnr', psnr.item ())
			train_vis.img ('LR image', inputs.cpu ().detach ().numpy ())
			train_vis.img ('HR image', fake_imgs.mul (255).cpu ().detach ().numpy ())
			train_vis.img ('GT image', real_imgs.cpu ().detach ().numpy ())

			# train results：
			# print('fake_out:{} real_out:{} L1_loss:{}'.format (fake_out, real_out, L1_loss (fake_imgs, real_imgs),edge_loss (fake_imgs, real_imgs)))
			print('epoch:[{}/{}] batch:[{}/{}] g_loss:{:.10f} d_loss:{:.10f} psnr:{:.10f}'.format(epoch, epochs, iteration, len(dataloader), g_loss.item(), d_loss.item(), psnr))

			if iteration % interval == 0:
				idx = np.random.choice (opt.batch_size)
				sr = tensor_to_np(fake_imgs[idx])
				gt = tensor_to_np(real_imgs[idx])
				fig = cv2.hconcat((sr, gt))
				save_train_img(fig, save_img_dir, 'fig', epoch, iteration)

# testing code
def test_process(test_dataloader, generator, save_img_dir):
	for idx, (inputs, labels) in enumerate(test_dataloader):
		inputs = Variable(inputs)
		labels = Variable(labels)
		if opt.cuda:
			inputs = inputs.cuda()
			labels = labels.cuda()
		prediction = generator(inputs)
		psnr = PSNR(prediction, labels)
		i = np.random.choice (opt.batch_size)
		sr = tensor_to_np (prediction[i])
		gt = tensor_to_np (labels[i])
		fig = cv2.hconcat ((sr, gt))
		save_test_img (fig, save_img_dir, 'test_fig', idx)
		print('batch{} ==> psnr:{}'.format(idx, psnr))

# adjustable learning rate
def adjust_learning_rate(epoch):
	lr = opt.lr * (0.1 ** (epoch // opt.step))
	if lr < 1e-6:
		lr = 1e-6
	return lr


loader = transforms.Compose ([transforms.ToTensor ()])
unloader = transforms.ToPILImage ()


def tensor_to_np(tensor):
	img = tensor.mul (255).byte ()
	img = img.cpu ().numpy ().squeeze ()
	return img


def save_train_img(image, image_dir, img_name, epoch, iteration):
	if not os.path.exists (image_dir):
		os.mkdir (image_dir)
	image_path = os.path.join (image_dir, img_name + '{}_{}.png'.format (epoch, iteration))
	sio.imsave (image_path, image)

def save_test_img(image, image_dir, img_name, iteration):
	if not os.path.exists (image_dir):
		os.mkdir (image_dir)
	image_path = os.path.join (image_dir, img_name + '{}.png'.format (iteration))
	sio.imsave (image_path, image)

def PSNR(pred, gt):
	pred = pred.cpu ().detach ()
	gt = gt.cpu ().detach ()
	pred = pred.clamp (0, 1)
	diff = pred - gt
	mse = np.mean (diff.numpy () ** 2)
	if mse == 0:
		return 100
	return 10 * np.log10 (1.0 / mse)


def save_checkpoint(generator, discriminator, epoch):
	model_folder = "model_para/"
	gen_parm_path = model_folder + "gen_parm_epoch{}.pkl".format(epoch)
	dis_parm_path = model_folder + "dis_parm_epoch{}.pkl".format(epoch)
	gen_state = {"epoch": epoch, "model": generator}
	dis_state = {"epoch": epoch, "model": discriminator}
	if not os.path.exists(model_folder):
		os.makedirs(model_folder)
	torch.save(gen_state, gen_parm_path)
	torch.save(dis_state, dis_parm_path)
	print("Checkpoint saved to {}, {}".format(gen_parm_path, dis_parm_path))


# pretained
if opt.pretrained_gen:  # training completed
	# conduct testing procedure
	print('==>loading test data...')
	test_dataset = get_dataset(opt.LR_test_dir, opt.HR_test_dir, opt.img_height, opt.img_width, opt.scale_factor)
	test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
	if os.path.isfile(opt.pretrained_gen):
		print('==> loading model {}'.format(opt.pretrained_gen))
		gen_weights = torch.load(opt.pretrained_gen)
		generator.load_state_dict(gen_weights['model'].state_dict())
		test(test_dataloader, generator, './test_res')
	else:
		print('==> no generator model found at {}'.format(opt.pretrained_gen))
else:  # not fully trained
	# conduct training procedure
	print('==>loading training data...')
	train_dataset = get_dataset(opt.LR_train_dir, opt.HR_train_dir, opt.img_height, opt.img_width, opt.scale_factor)
	train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.threads)
	if opt.resume_gen and opt.resume_dis:  # partly trained
		if os.path.isfile(opt.resume_gen and opt.resume_dis):
			gen_checkpoint = torch.load(opt.resume_gen)
			dis_checkpoint = torch.load(opt.resume_dis)
			opt.start_epoch = gen_checkpoint['epoch'] + 1
			print('==>start training at epoch {}'.format(opt.start_epoch))
			generator.load_state_dict(gen_checkpoint['model'].state_dict())
			discriminator.load_state_dict(dis_checkpoint['model'].state_dict())
			print("===> resume Training...")
			# resume training procedure
			train(train_dataloader, generator, discriminator, Gen_optim, Dis_optim, content_loss, pixel_loss, './train_res')
		else:
			print('==> cannot start training at epoch {}'.format(opt.start_epoch))
	else: # training from scratch
		train(train_dataloader, generator, discriminator, Gen_optim, Dis_optim, content_loss, pixel_loss, './train_res')
