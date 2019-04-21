import argparse
import os
from PIL import Image
import numpy as np
from torchvision import transforms
import skimage.measure as sm
from torchvision.transforms import Compose, ToTensor, Resize
import cv2

parser = argparse.ArgumentParser (description='eval')
parser.add_argument('--falg', type=bool, default=False, help='whether ROI selected')
parser.add_argument ('--img_height', type=int, default=448, help='height of HR images')
parser.add_argument ('--img_width', type=int, default=896, help='width of HR images')
parser.add_argument('--saved_ROIs_file', type=str, default='./ROIs.npz')
parser.add_argument('--num_sig_ROIs', type=int, default=4)
parser.add_argument ('--SR_image_dir', type=str, default='./our_res/x2')
parser.add_argument ('--LR_image_dir', type=str, default='./test_imgs/test1')
parser.add_argument ('--GT_image_dir', type=str, default='./test_imgs/average1')
parser.add_argument('--save_dir', type=str, default='./our_res/x2_ROI_res/')
parser.add_argument('--selected', type=bool, default=False)
opt = parser.parse_args ()


# print(len(fig1_sig_region))

def crop(img):
	return transforms.CenterCrop ((448, 896)) (img)


def is_image_file(filename):
	return any (filename.endswith (extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".tif"])


def load_img(filepath):
	y = Image.open (filepath).convert ('L')
	y = crop (y)
	return y


def LR_transform(img_height, img_width, scale_factor):
	return Compose ([Resize ((img_height // scale_factor, img_width // scale_factor), Image.BICUBIC), ToTensor ()])


def HR_transform():
	return Compose ([ToTensor ()])


SR_image_filenames = sorted ([os.path.join (opt.SR_image_dir, x) for x in os.listdir (opt.SR_image_dir) if is_image_file (x)])
GT_image_filenames = sorted ([os.path.join (opt.GT_image_dir, x) for x in os.listdir (opt.GT_image_dir) if is_image_file (x)])


def PSNR(X1, X2):
	X1 = np.asarray (X1)
	X2 = np.asarray (X2)
	psnr = sm.compare_psnr (X2, X1)
	return psnr

def SSIM(X1, X2):
	X1 = np.asarray (X1)
	X2 = np.asarray (X2)
	ssim = sm.compare_ssim(X1, X2, 11)
	return ssim


def EPI(X1, X2):
	X1 = np.asarray (X1)
	X2 = np.asarray (X2)
	X1_up = X1[:-1, :]
	X1_down = X1[1:, :]
	X2_up = X2[:-1, :]
	X2_down = X2[1:, :]
	return int (np.sum (np.abs (X1_up - X1_down))) / int (np.sum (np.abs (X2_up - X2_down)))


# ------------------------------------------------------------------------------------------------------------
# SNR1: 度量图像相似度
# 		如果把f(x,y)看做原始图g(x,y)和噪声图像e(x,y)的和，输出图的均方信噪比就是:
# 		sum(f(x, y)^2) / sum((f(x, y) - g(x, y))^2)
# SNR2: 度量噪声强度
# 		通常我们以平均值代表影像图像信号的强度，而将选定背景区域的标准差当成噪声的大小，
# 		因为标准差所代表的物理意义为噪声相对该区域平均值的波动情形，这直觉上就像是噪声載在图像信号上一樣，
# 		SNR越大，代表图像品质越好。常见的度量噪声强度的SNR的表达式为：
# 		SNR = 10log10(max(I)^2 / sigma^2)
# 		其中I表示图像，max（I）就是求取图像的最大像素，分子则是选定的背景噪声区域的方差（没有根号就是标准差）
# ------------------------------------------------------------------------------------------------------------
def SNR1(img_pred, img_true):
	img_pred = np.asarray(img_pred)
	img_true = np.asarray(img_true)
	err = img_true - img_pred
	snr1 = np.sum(np.power(img_pred, 2)) / np.sum(np.power(err, 2))
	return snr1

def SNR2(img_pred, bg_region):
	img_pred = np.asarray(img_pred)
	bg_ROI = img_pred[bg_region[1]:bg_region[3], bg_region[0]:bg_region[2]]
	snr2 = 10*np.log10(np.power(np.max(img_pred), 2) / np.power(np.std(bg_ROI), 2))
	return snr2


# ----------------------------------------------------
# 第i个ROI的等视数（equivalent numbers of looks）
# 等效视数（ENL），这是一种衡量均匀区域的光滑性的指标
# ENL_i = miu(i)^2 / sigma(i)^2
# ----------------------------------------------------
def ENL1(img, sig_region):
	enl = 0
	img = np.asarray(img)
	for i in range (len (sig_region)):
		top, bottom, left, right = sig_region[i][1], sig_region[i][3], sig_region[i][0], sig_region[i][2]
		ROI = img[top:bottom, left:right]
		miu = np.mean(ROI)
		sigma = np.std(ROI)
		enl += miu ** 2 / sigma ** 2
	enl = enl / len(sig_region)
	return enl

def ENL2(img, bg_region):
	img = np.asarray(img)
	top, bottom, left, right = bg_region[1], bg_region[3], bg_region[0], bg_region[2]
	ROI = img[top:bottom, left:right]
	miu = np.mean(ROI)
	sigma = np.std(ROI)
	print('miu:',miu)
	print('sigma:', sigma)
	enl = miu ** 2 / sigma ** 2
	return enl

# -----------------------------------------------------------------------------------------------------------------
# CNR: 对比度噪声比，是衡量图像对比度的一个指标。运用他的时候，首先要在待处理图像上选取感兴趣区域(ROI).
# CNR= 图像在感兴趣的区域内外的强度差 除以 图像在感兴趣的区域之内外的标准差和，并取绝对值。
# 即內外信号强度初一内外噪声和。可以很直观的想象，CNR的值越大代表內外区域越能被分辨出來，表示影像的对比度越好。
# ----------------------------------------------------------------------------------------------------------------
def CNR(img, bg_region, sig_region):
	cnr = 0
	img = np.asarray(img)
	bg_ROI = img[bg_region[1]:bg_region[3], bg_region[0]:bg_region[2]]
	bg_mean = np.mean(bg_ROI)
	bg_std = np.std(bg_ROI)
	for i in range(len(sig_region)):
		top, bottom, left, right = sig_region[i][1], sig_region[i][3], sig_region[i][0], sig_region[i][2]
		sig_ROI = img[top:bottom, left:right]
		cnr += 10*np.log10((np.mean(sig_ROI) - bg_mean) / np.sqrt(bg_std ** 2 + np.std(sig_ROI) ** 2))
	cnr = cnr / len(sig_region)
	return cnr

def MSR(img, sig_region):
	msr = 0
	img = np.asarray (img)
	for i in range(len(sig_region)):
		top, bottom, left, right = sig_region[i][1], sig_region[i][3], sig_region[i][0], sig_region[i][2]
		sig_ROI = img[top:bottom, left:right]
		miu = np.mean(sig_ROI)
		sigma = np.std(sig_ROI)
		msr += miu / sigma
	msr = msr / len(sig_region)
	return msr


def draw_roi(img, bg_region, fig_sig_region, img_num, img_dir):
	img = img.convert('RGB')
	img = np.asarray(img)
	cv2.rectangle(img, (bg_region[0], bg_region[1]), (bg_region[2], bg_region[3]), (0, 255, 0), thickness=2)
	for i in range(len(fig_sig_region)):
		cv2.rectangle(img, (fig_sig_region[i][0], fig_sig_region[i][1]), (fig_sig_region[i][2], fig_sig_region[i][3]), (255, 0, 0), thickness=2)
	if not os.path.exists(opt.save_dir):
		os.mkdir(opt.save_dir)
	img_path = os.path.join(opt.save_dir, img_dir)
	if not os.path.exists(img_path):
		os.mkdir(img_path)
	img = Image.fromarray(img)
	img.save(img_path + '/{}.tif'.format(img_num))


def load_ROIs(file_name):
	ROIs = np.loadtxt(file_name, int)
	bg_region = ROIs[0:]
	sig_regions = []
	for i in range(len(ROIs)):
		sig_regions.append(ROIs[i+1:])
	return bg_region, sig_regions


def select_ROIs(num_sig_ROIs):
	figs_bg_regions = []	# 所有图的背景区域的ROI
	figs_sig_regions = []	# 所有图像的信号区域的ROI
	for name in sorted(os.listdir(opt.SR_image_dir)):
		SR_img_path = os.path.join(opt.SR_image_dir, name)
		SR_img = np.asarray (load_img (SR_img_path))
		print(name)
		showCrosshair = False
		fromCenter = False
		# select background ROI
		print("===>Selet a background ROI")
		bg_roi = cv2.selectROI ("bg_ROI", SR_img, fromCenter, showCrosshair)
		bg_region = [int (bg_roi[0]), int (bg_roi[1]), int (bg_roi[0] + bg_roi[2]), int (bg_roi[1] + bg_roi[3])]

		# select signal ROIs
		sig_regions = []
		print ("===>Selet {} background ROIs".format(opt.num_sig_ROIs))
		for i in range (num_sig_ROIs):
			sig_roi = cv2.selectROI ("sig_ROI", SR_img, fromCenter, showCrosshair)
			sig_region = [int (sig_roi[0]), int (sig_roi[1]), int (sig_roi[0] + sig_roi[2]), int (sig_roi[1] + sig_roi[3])]
			# save signal ROI region
			sig_regions.append (sig_region)
		figs_bg_regions.append(bg_region)
		figs_sig_regions.append(sig_regions)
	# 保存得到的ROI区域位置坐标
	save_ROI_regions(figs_bg_regions, figs_sig_regions)
	return figs_bg_regions, figs_sig_regions


def crop_save_ROI(img, roi_save_dir, bg_region, sig_regions, img_num):
	img = np.asarray(img)
	if not os.path.exists(roi_save_dir):
		os.mkdir(roi_save_dir)
	bg_img = img[bg_region[1]:bg_region[3], bg_region[0]:bg_region[2]]
	bg_img_path = os.path.join (roi_save_dir, 'bgROI{}.tif'.format (img_num))
	cv2.imwrite(bg_img_path, bg_img)
	for i in range(len(sig_regions)):
		sig_img = img[sig_regions[i][1]:sig_regions[i][3], sig_regions[i][0]:sig_regions[i][2]]
		sig_img_path = os.path.join (roi_save_dir, 'sigROI{}_{}.tif'.format (img_num, i))
		cv2.imwrite(sig_img_path, sig_img)


def save_ROI_regions(figs_bg_regions, figs_sig_regions):
	figs_bg_regions = np.asarray (figs_bg_regions)
	figs_sig_regions = np.asarray (figs_sig_regions)
	np.savez (opt.saved_ROIs_file, figs_bg_regions, figs_sig_regions)


def load_ROI_regions():
	file = np.load(opt.saved_ROIs_file)
	figs_bg_regions, figs_sig_regions = file['arr_0'], file['arr_1']
	print (figs_bg_regions.tolist())
	print (figs_sig_regions.tolist())
	return figs_bg_regions, figs_sig_regions


def save_results(figs_bg_regions, figs_sig_regions):
	f1 = open ('./psnr.txt', 'w', encoding='UTF-8', errors='ignore')
	f2 = open ('./epi.txt', 'w', encoding='UTF-8', errors='ignore')
	f3 = open ('./snr1.txt', 'w', encoding='UTF-8', errors='ignore')
	f4 = open ('./snr2.txt', 'w', encoding='UTF-8', errors='ignore')
	f5 = open ('./cnr.txt', 'w', encoding='UTF-8', errors='ignore')
	f6 = open ('./enl.txt', 'w', encoding='UTF-8', errors='ignore')
	f7 = open ('./ssim.txt', 'w', encoding='UTF-8', errors='ignore')
	f8 = open ('./msr.txt', 'w', encoding='UTF-8', errors='ignore')

	total_psnr = 0
	total_epi = 0
	total_enl = 0
	total_cnr = 0
	total_snr2 = 0
	total_snr1 = 0
	total_ssim = 0
	total_msr = 0

	N = 1

	roi_dir = os.path.join (opt.save_dir, './croped_ROI')
	noisy_dir = os.path.join (opt.save_dir, './noise_roi')
	label_dir = os.path.join (opt.save_dir, './label_roi')

	for name in sorted(os.listdir(opt.GT_image_dir)):
		SR_img_path = os.path.join(opt.SR_image_dir, name)
		LR_img_path = os.path.join(opt.LR_image_dir, name)
		GT_img_path = os.path.join (opt.GT_image_dir, name)
		SR_img = crop(load_img (SR_img_path))
		GT_img = crop(load_img (GT_img_path))
		LR_img = crop (load_img (LR_img_path))

		if N == 1 and N == 14:
			N += 1

		img_num = int(name.split(".")[0])
		print(img_num)
		bg_region = figs_bg_regions[N-1]
		# print(bg_region)
		sig_regions = figs_sig_regions[N-1]
		# print(sig_regions)
		draw_roi(SR_img, bg_region, sig_regions, img_num, 'SR')
		draw_roi (GT_img, bg_region, sig_regions, img_num, 'GT')
		draw_roi (LR_img, bg_region, sig_regions, img_num, 'LR')

		crop_save_ROI(SR_img, roi_dir, bg_region, sig_regions, img_num)
		crop_save_ROI(GT_img, label_dir, bg_region, sig_regions, img_num)
		crop_save_ROI (LR_img, noisy_dir, bg_region, sig_regions, img_num)

		# PSNR
		psnr = PSNR (SR_img, GT_img)
		f1.write (str (psnr) + '\n')

		# EPI
		epi = EPI (SR_img, GT_img)
		f2.write (str (epi) + '\n')

		# SNR1
		snr1 = SNR1(SR_img, GT_img)
		f3.write(str (snr1) + '\n')

		# SNR2
		snr2 = SNR2 (SR_img, bg_region)
		f4.write (str (snr2) + '\n')

		# CNR
		cnr = CNR(SR_img, bg_region, sig_regions)
		f5.write (str (cnr) + '\n')

		# ENL
		enl = ENL2(SR_img, bg_region)
		f6.write (str (enl) + '\n')


		# SSIM
		ssim = SSIM(SR_img, GT_img)
		f7.write (str (ssim) + '\n')

		# MSR
		msr = MSR (SR_img, sig_regions)
		f8.write (str (msr) + '\n')

		total_psnr += psnr
		total_epi += epi
		total_enl += enl
		total_cnr += cnr
		total_snr1 += snr1
		total_snr2 += snr2
		total_ssim += ssim
		total_msr += msr

		N += 1
		print(N)
	aver_psnr = total_psnr / 16
	aver_epi = total_epi / 16
	aver_enl = total_enl / 16
	aver_snr2 = total_snr2 / 16
	aver_cnr = total_cnr / 16
	aver_snr1 = total_snr1 / 16
	aver_ssim = total_ssim / 16
	aver_msr = total_msr / 16

	print('aver_psnr:', aver_psnr)
	print('aver_epi:', aver_epi)
	print('aver_enl:', aver_enl)
	print('aver_cnr:', aver_cnr)
	print('aver_snr1:', aver_snr1)
	print('aver_snr2:', aver_snr2)
	print('aver_ssim:', aver_ssim)
	print('aver_msr:', aver_msr)

	f1.close ()
	f2.close ()
	f3.close ()
	f4.close()
	f5.close()
	f6.close()
	f7.close()
	f8.close()


if __name__ == '__main__':
	if opt.selected == False:
		figs_bg_regions, figs_sig_regions = select_ROIs(opt.num_sig_ROIs)
	else:
		figs_bg_regions, figs_sig_regions = load_ROI_regions()
	save_results (figs_bg_regions, figs_sig_regions)