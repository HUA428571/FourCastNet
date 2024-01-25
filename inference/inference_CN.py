# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os
import sys
import time
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from numpy.core.numeric import False_
import h5py
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, \
	weighted_acc_masked_torch_channels

logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime

# `fld` 是字段的名称，用于指定数据集中的特定字段。
# 不同字段（flds）有不同的decorrelation（装饰）时间，因此会有不同的初始条件（ics）。
fld = "z500"  # diff flds have diff decor times and hence differnt ics
# 判断fld的值，根据不同的值设置不同的DECORRELATION_TIME（装饰时间）
if fld == "z500" or fld == "2m_temperature" or fld == "t850":
    # 如果fld为z500、2m_temperature或t850，则DECORRELATION_TIME设置为36（对应9天）
    DECORRELATION_TIME = 36  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
else:
    # 如果fld为其他值，则DECORRELATION_TIME设置为8（对应2天）
    DECORRELATION_TIME = 8  # 9 days (36) for z500, 2 (8 steps) days for u10, v10
# idxes是一个字典，存储不同字段的索引，用于后续操作中快速定位这些字段。
idxes = {"u10": 0, "z500": 14, "2m_temperature": 2, "v10": 1, "t850": 5}


# 定义一个函数，用于给输入数据x添加高斯噪声，以增加数据的多样性。
def gaussian_perturb(x, level=0.01, device=0):
    # 根据输入数据x的形状，生成高斯分布的噪声。
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    # 将噪声加到输入数据x上，并返回结果。
    return (x + noise)


# 定义一个函数，用于加载模型，并从checkpoint文件中恢复模型的参数。
def load_model(model, params, checkpoint_file):
    # 清除模型中的梯度信息。
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    # 加载checkpoint文件。
    checkpoint = torch.load(checkpoint_fname)
    try:
        # 尝试读取checkpoint中的模型状态，并更新到当前模型中。
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:
        # 如果上述操作失败，则直接加载整个模型状态。
        model.load_state_dict(checkpoint['model_state'])
    # 将模型设为评估模式。
    model.eval()
    return model


# 定义一个函数，用于下采样输入数据x，减少数据的分辨率。
def downsample(x, scale=0.125):
    # 使用双线性插值方式对输入数据x进行下采样，并返回结果。
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')


# 定义一个函数，用于设置模型及相关参数，准备模型的训练或验证。
def setup(params):
    # 确定计算设备，优先使用GPU（如果可用）。
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    # 获取数据加载器，用于加载验证数据集。
    valid_data_loader, valid_dataset = get_data_loader(params, params.inf_data_path, dist.is_initialized(), train=False)
    # 从数据集中获取图片的尺寸信息，并更新到params中。
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    # 如果参数中设置了日志记录，则打印加载模型的信息。
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    # 从参数中获取输入和输出通道的信息，并计算它们的数量。
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)

    # 根据是否有地形（orography）信息，更新输入通道的数量。
    if params["orography"]:
        params['N_in_channels'] = n_in_channels + 1
    else:
        params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    # 加载全局平均值和标准差，用于数据的标准化处理。
    params.means = np.load(params.global_means_path)[0, out_channels]  # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]

    # 根据网络类型（nettype）加载相应的模型。
    if params.nettype == 'afno':
        model = AFNONet(params).to(device)
    else:
        raise Exception("not implemented")

    # 加载模型的checkpoint文件。
    checkpoint_file = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    # 加载用于验证的数据。
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    # 指定年份，用于从数据文件中选择特定的数据集。
    yr = 0
    if params.log_to_screen:
        logging.info('Loading inference data')
        logging.info('Inference data from {}'.format(files_paths[yr]))

    # 从指定的数据文件中加载验证数据。
    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']

    return valid_data_full, model

def autoregressive_inference(params, ic, valid_data_full, model):
    ic = int(ic)
    # 初始化全局变量
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'  # 确定使用的设备（GPU或CPU）
    exp_dir = params['experiment_dir']  # 实验目录
    dt = int(params.dt)  # 时间步长
    prediction_length = int(params.prediction_length / dt)  # 预测长度
    n_history = params.n_history  # 历史数据长度
    img_shape_x = params.img_shape_x  # 图像宽度
    img_shape_y = params.img_shape_y  # 图像高度
    in_channels = np.array(params.in_channels)  # 输入通道
    out_channels = np.array(params.out_channels)  # 输出通道
    n_in_channels = len(in_channels)  # 输入通道数
    n_out_channels = len(out_channels)  # 输出通道数
    means = params.means  # 数据平均值
    stds = params.stds  # 数据标准差

    # 初始化用于存储图像序列和RMSE/ACC的内存
    valid_loss = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)  # 验证损失
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)  # 准确率

    # 如果params.interp非零，则在粗分辨率下计算指标
    valid_loss_coarse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)  # 粗分辨率下的验证损失
    acc_coarse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)  # 粗分辨率下的准确率
    acc_coarse_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)  # 粗分辨率下的未加权准确率

    acc_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)  # 未加权准确率
    seq_real = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)  # 真实序列
    seq_pred = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)  # 预测序列

    acc_land = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)  # 陆地准确率
    acc_sea = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)  # 海洋准确率
    if params.masked_acc:
        maskarray = torch.as_tensor(np.load(params.maskpath)[0:720]).to(device, dtype=torch.float)  # 加载用于计算掩码准确率的掩码数组

    # 从第一年的数据中提取有效数据
    valid_data = valid_data_full[ic:(ic + prediction_length * dt + n_history * dt):dt, in_channels, 0:720]
    # 标准化数据
    valid_data = (valid_data - means) / stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    # 加载时间平均值
    if not params.use_daily_climatology:
        # 加载气候学时间平均值
        m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels] - means) / stds)[:, 0:img_shape_x]
        m = torch.unsqueeze(m, 0)
    else:
        # 加载每日气候学数据
        dc_path = params.dc_path
        with h5py.File(dc_path, 'r') as f:
            dc = f['time_means_daily'][ic:ic + prediction_length * dt:dt]  # 1460,21,721,1440
        m = torch.as_tensor((dc[:, out_channels, 0:img_shape_x, :] - means) / stds)

    m = m.to(device, dtype=torch.float)
    if params.interp > 0:
        m_coarse = downsample(m, scale=params.interp)  # 对时间平均值进行下采样

    std = torch.as_tensor(stds[:, 0, 0]).to(device, dtype=torch.float)

    orography = params.orography  # 地形信息
    orography_path = params.orography_path
    if orography:
        # 加载地形数据
        orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis=0), axis=0)).to(device, dtype=torch.float)
        logging.info("orography loaded; shape:{}".format(orog.shape))

    # 开始自回归推理
    if params.log_to_screen:
        logging.info('Begin autoregressive inference')

    with torch.no_grad():  # 不计算梯度，以加速推理过程并节省内存
        for i in range(valid_data.shape[0]):
            # 核心自回归推理代码块
            if i == 0:  # 序列的开始
                first = valid_data[0:n_history + 1]  # 提取历史数据
                future = valid_data[n_history + 1]  # 提取未来数据
                for h in range(n_history + 1):
                    # 提取历史数据中的每一步，并将其放入seq_real和seq_pred中
                    seq_real[h] = first[h * n_in_channels: (h + 1) * n_in_channels][0:n_out_channels]
                    seq_pred[h] = seq_real[h]
                if params.perturb:
                    # 如果需要，对初始条件进行扰动
                    first = gaussian_perturb(first, level=params.n_level, device=device)
                if orography:
                    # 如果有地形信息，将地形数据和历史数据一起输入模型
                    future_pred = model(torch.cat((first, orog), axis=1))
                else:
                    # 没有地形信息，只输入历史数据
                    future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    # 对于非序列末尾的步骤，提取下一步的未来数据
                    future = valid_data[n_history + i + 1]
                if orography:
                    # 如果有地形信息，将地形数据和前一步的预测结果一起输入模型
                    future_pred = model(torch.cat((future_pred, orog), axis=1))
                else:
                    # 没有地形信息，只输入前一步的预测结果
                    future_pred = model(future_pred)

            if i < prediction_length - 1:  # 不是最后一步
                # 更新seq_pred和seq_real
                seq_pred[n_history + i + 1] = future_pred
                seq_real[n_history + i + 1] = future
                # 准备下一步的输入数据
                history_stack = seq_pred[i + 1:i + 2 + n_history]

            future_pred = history_stack

            # 计算指标
            if params.use_daily_climatology:
                # 如果使用每日气候学数据，提取相应的时间平均值
                clim = m[i:i + 1]
                if params.interp > 0:
                    clim_coarse = m_coarse[i:i + 1]
            else:
                # 如果使用普通气候学数据，使用整体的时间平均值
                clim = m
                if params.interp > 0:
                    clim_coarse = m_coarse

            # 计算预测的误差和准确率
            pred = torch.unsqueeze(seq_pred[i], 0)  # 当前预测
            tar = torch.unsqueeze(seq_real[i], 0)  # 当前真实值
            valid_loss[i] = weighted_rmse_torch_channels(pred, tar) * std  # 计算加权RMSE
            acc[i] = weighted_acc_torch_channels(pred - clim, tar - clim)  # 计算加权ACC
            acc_unweighted[i] = unweighted_acc_torch_channels(pred - clim, tar - clim)  # 计算未加权ACC

            if params.masked_acc:
                # 如果需要计算掩码ACC，使用提供的掩码数据
                acc_land[i] = weighted_acc_masked_torch_channels(pred - clim, tar - clim, maskarray)
                acc_sea[i] = weighted_acc_masked_torch_channels(pred - clim, tar - clim, 1 - maskarray)

            if params.interp > 0:
                # 如果有粗分辨率指标，对预测和真实值进行下采样，并计算相应的指标
                pred = downsample(pred, scale=params.interp)
                tar = downsample(tar, scale=params.interp)
                valid_loss_coarse[i] = weighted_rmse_torch_channels(pred, tar) * std
                acc_coarse[i] = weighted_acc_torch_channels(pred - clim_coarse, tar - clim_coarse)
                acc_coarse_unweighted[i] = unweighted_acc_torch_channels(pred - clim_coarse, tar - clim_coarse)

            if params.log_to_screen:
                # 如果需要，打印日志信息
                idx = idxes[fld]
                logging.info('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, fld, valid_loss[i, idx],
                                                                                             acc[i, idx]))
                if params.interp > 0:
                    logging.info('[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, fld,
                                                                                                          valid_loss_coarse[i, idx],
                                                                                                          acc_coarse[i, idx]))

    # 将所有的结果从设备（GPU/CPU）转移到CPU，并转换为NumPy数组
    seq_real = seq_real.cpu().numpy()
    seq_pred = seq_pred.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    acc = acc.cpu().numpy()
    acc_unweighted = acc_unweighted.cpu().numpy()
    acc_coarse = acc_coarse.cpu().numpy()
    acc_coarse_unweighted = acc_coarse_unweighted.cpu().numpy()
    valid_loss_coarse = valid_loss_coarse.cpu().numpy()
    acc_land = acc_land.cpu().numpy()
    acc_sea = acc_sea.cpu().numpy()

    # 返回所有计算得到的指标和序列
    return (np.expand_dims(seq_real[n_history:], 0), np.expand_dims(seq_pred[n_history:], 0), np.expand_dims(valid_loss, 0),
            np.expand_dims(acc, 0), np.expand_dims(acc_unweighted, 0), np.expand_dims(valid_loss_coarse, 0), np.expand_dims(acc_coarse, 0),
            np.expand_dims(acc_coarse_unweighted, 0), np.expand_dims(acc_land, 0), np.expand_dims(acc_sea, 0))


if __name__ == '__main__':
	# 初始化命令行参数解析器
	parser = argparse.ArgumentParser()
	# 定义命令行参数
	parser.add_argument("--run_num", default='00', type=str)
	parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
	parser.add_argument("--config", default='full_field', type=str)
	parser.add_argument("--use_daily_climatology", action='store_true')
	parser.add_argument("--vis", action='store_true')
	parser.add_argument("--override_dir", default=None, type=str, help='Path to store inference outputs; must also set --weights arg')
	parser.add_argument("--interp", default=0, type=float)
	parser.add_argument("--weights", default=None, type=str, help='Path to model weights, for use with override_dir option')

	# 解析命令行参数
	args = parser.parse_args()

	# 从YAML文件加载参数
	params = YParams(os.path.abspath(args.yaml_config), args.config)
	# 设置全局参数
	params['world_size'] = 1
	params['interp'] = args.interp
	params['use_daily_climatology'] = args.use_daily_climatology
	params['global_batch_size'] = params.batch_size

	# 设置CUDA设备
	torch.cuda.set_device(0)
	torch.backends.cudnn.benchmark = True
	# 是否进行可视化
	vis = args.vis

	# 根据命令行参数设置实验目录
	if args.override_dir is not None:
		assert args.weights is not None, 'Must set --weights argument if using --override_dir'
		expDir = args.override_dir
	else:
		assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
		expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

	# 如果实验目录不存在，则创建它
	if not os.path.isdir(expDir):
		os.makedirs(expDir)

	# 设置实验目录和checkpoint路径
	params['experiment_dir'] = os.path.abspath(expDir)
	params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir,
																									 'training_checkpoints/best_ckpt.tar')
	params['resuming'] = False
	params['local_rank'] = 0

	# 配置日志
	logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
	logging_utils.log_versions()
	params.log()

	# 根据fld字段设置每年的样本数量
	n_ics = params['n_initial_conditions']
	if fld == "z500" or fld == "t850":
		n_samples_per_year = 1336
	else:
		n_samples_per_year = 1460

	# 根据参数设置初始条件
	if params["ics_type"] == 'default':
		# 默认初始条件设置
		num_samples = n_samples_per_year - params.prediction_length
		stop = num_samples
		ics = np.arange(0, stop, DECORRELATION_TIME)
		if vis:
			# 如果进行可视化，只针对第一个初始条件
			ics = [0]
		n_ics = len(ics)
	elif params["ics_type"] == "datetime":
		# 根据日期字符串设置初始条件
		date_strings = params["date_strings"]
		ics = []
		if params.perturb:
			# 对于扰动，使用单个日期创建n_ics个扰动
			n_ics = params["n_perturbations"]
			date = date_strings[0]
			date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
			day_of_year = date_obj.timetuple().tm_yday - 1
			hour_of_day = date_obj.timetuple().tm_hour
			hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
			for ii in range(n_ics):
				ics.append(int(hours_since_jan_01_epoch / 6))
		else:
			# 根据日期字符串计算初始条件
			for date in date_strings:
				date_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
				day_of_year = date_obj.timetuple().tm_yday - 1
				hour_of_day = date_obj.timetuple().tm_hour
				hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
				ics.append(int(hours_since_jan_01_epoch / 6))
		n_ics = len(ics)

	logging.info("Inference for {} initial conditions".format(n_ics))

	# 准备文件标签
	try:
		autoregressive_inference_filetag = params["inference_file_tag"]
	except:
		autoregressive_inference_filetag = ""

	if params.interp > 0:
		autoregressive_inference_filetag += "_coarse"

	autoregressive_inference_filetag += "_" + fld + ""
	if vis:
		autoregressive_inference_filetag += "_vis"

	# 获取验证数据和模型
	valid_data_full, model = setup(params)

	# 初始化存储图像序列和RMSE/ACC的列表
	valid_loss = []
	valid_loss_coarse = []
	acc_unweighted = []
	acc = []
	acc_coarse = []
	acc_coarse_unweighted = []
	seq_pred = []
	seq_real = []
	acc_land = []
	acc_sea = []

	# 对每个初始条件进行自回归推理
	for i, ic in enumerate(ics):
		logging.info("Initial condition {} of {}".format(i + 1, n_ics))
		sr, sp, vl, a, au, vc, ac, acu, accland, accsea = autoregressive_inference(params, ic, valid_data_full, model)

		# 如果是第一个初始条件或列表为空，则直接赋值
		if i == 0 or len(valid_loss) == 0:
			seq_real = sr
			seq_pred = sp
			valid_loss = vl
			valid_loss_coarse = vc
			acc = a
			acc_coarse = ac
			acc_coarse_unweighted = acu
			acc_unweighted = au
			acc_land = accland
			acc_sea = accsea
		else:
			# 对于后续的初始条件，将结果追加到列表中
			valid_loss = np.concatenate((valid_loss, vl), 0)
			valid_loss_coarse = np.concatenate((valid_loss_coarse, vc), 0)
			acc = np.concatenate((acc, a), 0)
			acc_coarse = np.concatenate((acc_coarse, ac), 0)
			acc_coarse_unweighted = np.concatenate((acc_coarse_unweighted, acu), 0)
			acc_unweighted = np.concatenate((acc_unweighted, au), 0)
			acc_land = np.concatenate((acc_land, accland), 0)
			acc_sea = np.concatenate((acc_sea, accsea), 0)

	# 提取序列的维度信息
	prediction_length = seq_real[0].shape[0]
	n_out_channels = seq_real[0].shape[1]
	img_shape_x = seq_real[0].shape[2]
	img_shape_y = seq_real[0].shape[3]

	# 保存预测结果和损失到HDF5文件
	if params.log_to_screen:
		logging.info("Saving files at {}".format(
			os.path.join(params['experiment_dir'], 'autoregressive_predictions' + autoregressive_inference_filetag + '.h5')))
	with h5py.File(os.path.join(params['experiment_dir'], 'autoregressive_predictions' + autoregressive_inference_filetag + '.h5'),
				   'a') as f:
		# 存储数据的逻辑，包括可视化数据，地表和海面的准确率，以及不同分辨率下的RMSE和ACC
		if vis:
			try:
				f.create_dataset("ground_truth", data=seq_real, shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y),
								 dtype=np.float32)
			except:
				del f["ground_truth"]
				f.create_dataset("ground_truth", data=seq_real, shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y),
								 dtype=np.float32)
				f["ground_truth"][...] = seq_real

			try:
				f.create_dataset("predicted", data=seq_pred, shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y),
								 dtype=np.float32)
			except:
				del f["predicted"]
				f.create_dataset("predicted", data=seq_pred, shape=(n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y),
								 dtype=np.float32)
				f["predicted"][...] = seq_pred

		if params.masked_acc:
			try:
				f.create_dataset("acc_land", data=acc_land)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
			except:
				del f["acc_land"]
				f.create_dataset("acc_land", data=acc_land)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
				f["acc_land"][...] = acc_land

			try:
				f.create_dataset("acc_sea", data=acc_sea)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
			except:
				del f["acc_sea"]
				f.create_dataset("acc_sea", data=acc_sea)  # , shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
				f["acc_sea"][...] = acc_sea

		try:
			f.create_dataset("rmse", data=valid_loss, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
		except:
			del f["rmse"]
			f.create_dataset("rmse", data=valid_loss, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
			f["rmse"][...] = valid_loss

		try:
			f.create_dataset("acc", data=acc, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
		except:
			del f["acc"]
			f.create_dataset("acc", data=acc, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
			f["acc"][...] = acc

		try:
			f.create_dataset("rmse_coarse", data=valid_loss_coarse, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
		except:
			del f["rmse_coarse"]
			f.create_dataset("rmse_coarse", data=valid_loss_coarse, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
			f["rmse_coarse"][...] = valid_loss_coarse

		try:
			f.create_dataset("acc_coarse", data=acc_coarse, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
		except:
			del f["acc_coarse"]
			f.create_dataset("acc_coarse", data=acc_coarse, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
			f["acc_coarse"][...] = acc_coarse

		try:
			f.create_dataset("acc_unweighted", data=acc_unweighted, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
		except:
			del f["acc_unweighted"]
			f.create_dataset("acc_unweighted", data=acc_unweighted, shape=(n_ics, prediction_length, n_out_channels), dtype=np.float32)
			f["acc_unweighted"][...] = acc_unweighted

		try:
			f.create_dataset("acc_coarse_unweighted", data=acc_coarse_unweighted, shape=(n_ics, prediction_length, n_out_channels),
							 dtype=np.float32)
		except:
			del f["acc_coarse_unweighted"]
			f.create_dataset("acc_coarse_unweighted", data=acc_coarse_unweighted, shape=(n_ics, prediction_length, n_out_channels),
							 dtype=np.float32)
			f["acc_coarse_unweighted"][...] = acc_coarse_unweighted
		f.close()