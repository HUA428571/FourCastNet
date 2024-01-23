#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os
import time
import numpy as np
import argparse
import h5py
import torch
import cProfile
import re
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet, PrecipNet
from utils.img_utils import vis_precip
import wandb
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch, unlog_tp_torch
from apex import optimizers
from utils.darcy_loss import LpLoss
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
DECORRELATION_TIME = 36 # 9 days
import json
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

class Trainer():
	def count_parameters(self):
		return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

	def __init__(self, params, world_rank):
		# 初始化模型参数和训练设备
		self.params = params
		self.world_rank = world_rank
		self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

		# 如果配置了使用wandb（Weights & Biases），则初始化wandb日志
		if params.log_to_wandb:
			wandb.init(config=params, name=params.name, group=params.group, project=params.project, entity=params.entity)

		# 记录日志信息，表示数据加载器的初始化开始
		logging.info('rank %d, begin data loader init'%world_rank)
		# 初始化训练数据加载器
		self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path, dist.is_initialized(), train=True)
		# 初始化验证数据加载器
		self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)
		# 初始化损失函数
		self.loss_obj = LpLoss()
		# 记录日志信息，表示数据加载器已初始化
		logging.info('rank %d, data loader initialized'%world_rank)

		# 更新模型参数以匹配验证数据集的尺寸
		params.crop_size_x = self.valid_dataset.crop_size_x
		params.crop_size_y = self.valid_dataset.crop_size_y
		params.img_shape_x = self.valid_dataset.img_shape_x
		params.img_shape_y = self.valid_dataset.img_shape_y

   		# 检查是否训练降雨模型
   		self.precip = True if "precip" in params else False
	
		# 如果训练降雨模型
		if self.precip:
			# 检查是否提供了风模型的权重
			if 'model_wind_path' not in params:
				raise Exception("no backbone model weights specified")
			# 加载风模型，输入输出通道数相同
			out_channels = np.array(params['in_channels'])
			params['N_out_channels'] = len(out_channels)

			# 根据配置初始化风模型
			if params.nettype_wind == 'afno':
				self.model_wind = AFNONet(params).to(self.device)
			else:
				raise Exception("not implemented")

			# 如果使用分布式训练，应用分布式数据并行
			if dist.is_initialized():
				self.model_wind = DistributedDataParallel(self.model_wind, device_ids=[params.local_rank], output_device=[params.local_rank], find_unused_parameters=True)
			# 加载风模型权重
			self.load_model_wind(params.model_wind_path)
			# 关闭风模型的梯度计算
			self.switch_off_grad(self.model_wind)

		# 重置输出通道数为降雨模型的输出通道数
		if self.precip:
			params['N_out_channels'] = len(params['out_channels'])

		# 初始化主模型
		if params.nettype == 'afno':
			self.model = AFNONet(params).to(self.device) 
		else:
			raise Exception("not implemented")
	
		# 如果是降雨模型，使用PrecipNet作为主模型
		if self.precip:
			self.model = PrecipNet(params, backbone=self.model).to(self.device)

		# 如果启用NHWC内存格式，转换模型
		if self.params.enable_nhwc:
			self.model = self.model.to(memory_format=torch.channels_last)

		# 如果使用wandb，监视模型参数
		if params.log_to_wandb:
			wandb.watch(self.model)

		# 根据配置初始化优化器
		if params.optimizer_type == 'FusedAdam':
			self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr = params.lr)
		else:
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)

		# 如果启用自动混合精度训练，初始化渐变缩放器
		if params.enable_amp == True:
			self.gscaler = amp.GradScaler()

		# 如果使用分布式训练，应用分布式数据并行
		if dist.is_initialized():
			self.model = DistributedDataParallel(self.model, device_ids=[params.local_rank], output_device=[params.local_rank], find_unused_parameters=True)

		# 初始化迭代计数器和起始训练轮数
		self.iters = 0
		self.startEpoch = 0
		# 如果从检查点恢复训练，加载检查点
		if params.resuming:
			logging.info("Loading checkpoint %s"%params.checkpoint_path)
			self.restore_checkpoint(params.checkpoint_path)
		# 如果使用两步训练且未恢复训练，从预训练模型开始
		if params.two_step_training:
			if params.resuming == False and params.pretrained == True:
				logging.info("Starting from pretrained one-step afno model at %s"%params.pretrained_ckpt_path)
				self.restore_checkpoint(params.pretrained_ckpt_path)
				self.iters = 0
				self.startEpoch = 0

		# 设置当前训练轮数
		self.epoch = self.startEpoch
	
		# 根据配置初始化学习率调整器
		if params.scheduler == 'ReduceLROnPlateau':
			self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
		elif params.scheduler == 'CosineAnnealingLR':
			self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs, last_epoch=self.startEpoch-1)
		else:
			self.scheduler = None
	
		# 如果配置了屏幕日志，打印模型的可训练参数数量
		if params.log_to_screen:
			logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))


	def switch_off_grad(self, model):
		# 关闭模型的梯度计算
		# 参数:
		#   - model: 要关闭梯度计算的模型
		for param in model.parameters():
			param.requires_grad = False

	def train(self):
		# 如果配置了屏幕日志，打印开始训练的信息
		if self.params.log_to_screen:
			logging.info("Starting Training Loop...")

		# 初始化最佳验证损失为一个很大的数
		best_valid_loss = 1.e6
		# 开始训练循环
		for epoch in range(self.startEpoch, self.params.max_epochs):
			# 如果使用分布式训练，设置采样器的当前轮数
			if dist.is_initialized():
				self.train_sampler.set_epoch(epoch)

			start = time.time()
			# 进行一轮训练，并记录训练时间、数据加载时间和训练日志
			tr_time, data_time, train_logs = self.train_one_epoch()
			# 进行一轮验证，并记录验证时间和验证日志
			valid_time, valid_logs = self.validate_one_epoch()
			# 在最后一轮训练后，如果预测类型为'direct'，进行最终验证
			if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
				valid_weighted_rmse = self.validate_final()

			# 根据当前验证损失调整学习率
			if self.params.scheduler == 'ReduceLROnPlateau':
				self.scheduler.step(valid_logs['valid_loss'])
			elif self.params.scheduler == 'CosineAnnealingLR':
				self.scheduler.step()
				# 如果达到最大轮数，结束训练
				if self.epoch >= self.params.max_epochs:
					logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
					exit()

			# 如果使用wandb，记录当前学习率
			if self.params.log_to_wandb:
				for pg in self.optimizer.param_groups:
					lr = pg['lr']
				wandb.log({'lr': lr})

			# 如果是主进程，处理检查点保存
			if self.world_rank == 0:
				if self.params.save_checkpoint:
					# 每轮结束时保存检查点
					self.save_checkpoint(self.params.checkpoint_path)
					# 如果验证损失有所改善，保存最佳检查点
					if valid_logs['valid_loss'] <= best_valid_loss:
						self.save_checkpoint(self.params.best_checkpoint_path)
						best_valid_loss = valid_logs['valid_loss']

			# 如果配置了屏幕日志，打印每轮的时间和损失信息
			if self.params.log_to_screen:
				logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
				logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))

			# 如果是最后一轮且预测类型为'direct'，打印最终的验证RMSE（注释掉的部分）
			# if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
			#     logging.info('Final Valid RMSE: Z500- {}. T850- {}, 2m_T- {}'.format(valid_weighted_rmse[0], valid_weighted_rmse[1], valid_weighted_rmse[2]))

	def train_one_epoch(self):
		# 增加训练轮数
		self.epoch += 1
		tr_time = 0  # 训练时间
		data_time = 0  # 数据加载时间
		self.model.train()  # 设置模型为训练模式
		
		# 遍历训练数据
		for i, data in enumerate(self.train_data_loader, 0):
			self.iters += 1
			# 记录数据加载开始时间
			data_start = time.time()
			# 将输入和目标数据移至设备，并转换为浮点型
			inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)
			# 如果配置了地形数据和两步训练
			if self.params.orography and self.params.two_step_training:
				orog = inp[:,-2:-1]

			# 如果启用了NHWC内存格式，转换输入和目标数据
			if self.params.enable_nhwc:
				inp = inp.to(memory_format=torch.channels_last)
				tar = tar.to(memory_format=torch.channels_last)

			# 如果目标包含残差场，从输入中减去相应的部分
			if 'residual_field' in self.params.target:
				tar -= inp[:, 0:tar.size()[1]]
			# 更新数据加载时间
			data_time += time.time() - data_start

			# 记录训练开始时间
			tr_start = time.time()

			# 清除之前的梯度
			self.model.zero_grad()
			# 如果是两步训练
			if self.params.two_step_training:
				with amp.autocast(self.params.enable_amp):
					# 第一步训练
					gen_step_one = self.model(inp).to(self.device, dtype = torch.float)
					loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.N_out_channels])
					# 如果包含地形数据
					if self.params.orography:
						gen_step_two = self.model(torch.cat((gen_step_one, orog), axis=1)).to(self.device, dtype = torch.float)
					else:
						gen_step_two = self.model(gen_step_one).to(self.device, dtype = torch.float)
					# 第二步损失
					loss_step_two = self.loss_obj(gen_step_two, tar[:,self.params.N_out_channels:2*self.params.N_out_channels])
					# 总损失
					loss = loss_step_one + loss_step_two
			else:
				with amp.autocast(self.params.enable_amp):
					# 单步训练
					if self.precip:  # 如果是降雨模型
						with torch.no_grad():
							inp = self.model_wind(inp).to(self.device, dtype = torch.float)
						gen = self.model(inp.detach()).to(self.device, dtype = torch.float)
					else:
						gen = self.model(inp).to(self.device, dtype = torch.float)
					# 计算损失
					loss = self.loss_obj(gen, tar)

			# 如果启用了自动混合精度，使用梯度缩放
			if self.params.enable_amp:
				self.gscaler.scale(loss).backward()
				self.gscaler.step(self.optimizer)
				self.gscaler.update()
			else:
				# 反向传播和优化步骤
				loss.backward()
				self.optimizer.step()

			# 更新训练时间
			tr_time += time.time() - tr_start
		
			# 记录日志
			try:
				logs = {'loss': loss, 'loss_step_one': loss_step_one, 'loss_step_two': loss_step_two}
			except:
				logs = {'loss': loss}

			# 如果使用分布式训练，对日志数据进行归约操作
			if dist.is_initialized():
				for key in sorted(logs.keys()):
					dist.all_reduce(logs[key].detach())
					logs[key] = float(logs[key]/dist.get_world_size())

			# 如果使用wandb，记录日志
			if self.params.log_to_wandb:
				wandb.log(logs, step=self.epoch)

			# 返回训练时间、数据加载时间和日志
			return tr_time, data_time, logs

	def validate_one_epoch(self):
		# 设置模型为评估模式
		self.model.eval()
		# 限定验证批次数量，通常用于学习率调整
		n_valid_batches = 20
		# 检查是否支持所用的标准化方法
		if self.params.normalization == 'minmax':
			raise Exception("minmax normalization not supported")
		elif self.params.normalization == 'zscore':
			mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)

		# 初始化验证损失和其他统计量
		valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
		valid_loss, valid_l1, valid_steps = valid_buff[0].view(-1), valid_buff[1].view(-1), valid_buff[2].view(-1)
		valid_weighted_rmse = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)
		valid_weighted_acc = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)

		# 记录验证开始时间
		valid_start = time.time()

		# 从验证集中随机选择一个样本索引
		sample_idx = np.random.randint(len(self.valid_data_loader))
		with torch.no_grad():
			for i, data in enumerate(self.valid_data_loader, 0):
				if (not self.precip) and i >= n_valid_batches:
					break
				inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
				if self.params.orography and self.params.two_step_training:
					orog = inp[:,-2:-1]

				# 执行两步训练策略
				if self.params.two_step_training:
					gen_step_one = self.model(inp).to(self.device, dtype=torch.float)
					loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.N_out_channels])
					if self.params.orography:
						gen_step_two = self.model(torch.cat((gen_step_one, orog), axis=1)).to(self.device, dtype=torch.float)
					else:
						gen_step_two = self.model(gen_step_one).to(self.device, dtype=torch.float)
					loss_step_two = self.loss_obj(gen_step_two, tar[:,self.params.N_out_channels:2*self.params.N_out_channels])
					valid_loss += loss_step_one + loss_step_two
					valid_l1 += nn.functional.l1_loss(gen_step_one, tar[:,0:self.params.N_out_channels])
				else:
					# 对于非两步训练
					if self.precip:
						with torch.no_grad():
							inp = self.model_wind(inp).to(self.device, dtype=torch.float)
						gen = self.model(inp.detach())
					else:
						gen = self.model(inp).to(self.device, dtype=torch.float)
					valid_loss += self.loss_obj(gen, tar) 
					valid_l1 += nn.functional.l1_loss(gen, tar)

				valid_steps += 1.

				# 保存用于可视化的字段
				if (i == sample_idx) and (self.precip and self.params.log_to_wandb):
					fields = [gen[0,0].detach().cpu().numpy(), tar[0,0].detach().cpu().numpy()]

				# 处理降雨预测
				if self.precip:
					gen = unlog_tp_torch(gen, self.params.precip_eps)
					tar = unlog_tp_torch(tar, self.params.precip_eps)

				# 直接预测的加权均方根误差
				if self.params.two_step_training:
					if 'residual_field' in self.params.target:
						valid_weighted_rmse += weighted_rmse_torch((gen_step_one + inp), (tar[:,0:self.params.N_out_channels] + inp))
					else:
						valid_weighted_rmse += weighted_rmse_torch(gen_step_one, tar[:,0:self.params.N_out_channels])
				else:
					if 'residual_field' in self.params.target:
						valid_weighted_rmse += weighted_rmse_torch((gen + inp), (tar + inp))
					else:
						valid_weighted_rmse += weighted_rmse_torch(gen, tar)

				# 保存验证图片
				if not self.precip:
					try:
						os.mkdir(params['experiment_dir'] + "/" + str(i))
					except:
						pass
					if self.params.two_step_training:
						save_image(torch.cat((gen_step_one[0,0], torch.zeros((self.valid_dataset.img_shape_x, 4)).to(self.device, dtype=torch.float), tar[0,0]), axis=1), params['experiment_dir'] + "/" + str(i) + "/" + str(self.epoch) + ".png")
					else:
						save_image(torch.cat((gen[0,0], torch.zeros((self.valid_dataset.img_shape_x, 4)).to(self.device, dtype=torch.float), tar[0,0]), axis=1), params['experiment_dir'] + "/" + str(i) + "/" + str(self.epoch) + ".png")

		# 如果使用分布式训练，进行数据归约
		if dist.is_initialized():
			dist.all_reduce(valid_buff)
			dist.all_reduce(valid_weighted_rmse)

		# 除以步数以获取平均值
		valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
		valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
		if not self.precip:
			valid_weighted_rmse *= mult

		# 转换为CPU张量
		valid_buff_cpu = valid_buff.detach().cpu().numpy()
		valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()

		# 计算验证时间
		valid_time = time.time() - valid_start
		valid_weighted_rmse = mult * torch.mean(valid_weighted_rmse, axis=0)
		# 根据是否为降雨模型生成日志
		if self.precip:
			logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_tp': valid_weighted_rmse_cpu[0]}
		else:
			try:
				logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_u10': valid_weighted_rmse_cpu[0], 'valid_rmse_v10': valid_weighted_rmse_cpu[1]}
			except:
				logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_u10': valid_weighted_rmse_cpu[0]}

		# 如果使用wandb，记录验证日志
		if self.params.log_to_wandb:
			if self.precip:
				fig = vis_precip(fields)
				logs['vis'] = wandb.Image(fig)
				plt.close(fig)
			wandb.log(logs, step=self.epoch)

		return valid_time, logs


	def validate_final(self):
		# 将模型设置为评估模式
		self.model.eval()
		# 根据验证数据集的大小来决定验证批次的数量
		n_valid_batches = int(self.valid_dataset.n_patches_total/self.valid_dataset.n_patches)
		# 初始化一个用于存储加权均方根误差的数组
		valid_weighted_rmse = torch.zeros(n_valid_batches, self.params.N_out_channels)
		# 检查标准化方法，如果是minmax则抛出异常
		if self.params.normalization == 'minmax':
			raise Exception("minmax normalization not supported")
		elif self.params.normalization == 'zscore':
			# 加载全局标准差用于反标准化
			mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)

		with torch.no_grad():
			for i, data in enumerate(self.valid_data_loader):
				if i > 100:
					break
				inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
				if self.params.orography and self.params.two_step_training:
					orog = inp[:,-2:-1]
				if 'residual_field' in self.params.target:
					tar -= inp[:, 0:tar.size()[1]]

				# 如果是两步训练
				if self.params.two_step_training:
					gen_step_one = self.model(inp).to(self.device, dtype=torch.float)
					loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.N_out_channels])
					if self.params.orography:
						gen_step_two = self.model(torch.cat((gen_step_one, orog), axis=1)).to(self.device, dtype=torch.float)
					else:
						gen_step_two = self.model(gen_step_one).to(self.device, dtype=torch.float)
					loss_step_two = self.loss_obj(gen_step_two, tar[:,self.params.N_out_channels:2*self.params.N_out_channels])
					# 计算两步训练的总损失
					valid_loss = loss_step_one + loss_step_two
				else:
					# 如果是单步训练
					gen = self.model(inp)
					valid_loss = self.loss_obj(gen, tar)

				# 计算加权均方根误差
				for c in range(self.params.N_out_channels):
					if 'residual_field' in self.params.target:
						valid_weighted_rmse[i, c] = weighted_rmse_torch((gen[0,c] + inp[0,c]), (tar[0,c]+inp[0,c]), self.device)
					else:
						valid_weighted_rmse[i, c] = weighted_rmse_torch(gen[0,c], tar[0,c], self.device)

			# 反标准化
			valid_weighted_rmse = mult * torch.mean(valid_weighted_rmse[0:100], axis=0).to(self.device)

		return valid_weighted_rmse



	def load_model_wind(self, model_path):
		# 在屏幕上打印加载风模型权重的信息
		if self.params.log_to_screen:
			logging.info('Loading the wind model weights from {}'.format(model_path))
		# 加载模型权重
		checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.params.local_rank))
		# 如果是分布式训练，则直接加载模型状态字典
		if dist.is_initialized():
			self.model_wind.load_state_dict(checkpoint['model_state'])
		else:
			# 对于非分布式环境，需要重新整理模型状态字典
			new_model_state = OrderedDict()
			model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
			for key in checkpoint[model_key].keys():
				if 'module.' in key:  # 如果模型是使用数据并行保存的，则需要移除前缀
					name = key[7:]
					new_model_state[name] = checkpoint[model_key][key]
				else:
					new_model_state[key] = checkpoint[model_key][key]
			# 加载新的模型状态字典
			self.model_wind.load_state_dict(new_model_state)
			# 将模型设置为评估模式
			self.model_wind.eval()


	def save_checkpoint(self, checkpoint_path, model=None):
		# 如果没有指定模型，则使用当前模型
		if not model:
			model = self.model
		# 保存模型的迭代次数、训练轮数、模型状态和优化器状态
		torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
					'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)


	def restore_checkpoint(self, checkpoint_path):
		# 从指定路径加载检查点
		checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
		try:
			# 尝试直接加载模型状态
			self.model.load_state_dict(checkpoint['model_state'])
		except:
			# 如果直接加载失败，则需要调整模型状态字典
			new_state_dict = OrderedDict()
			for key, val in checkpoint['model_state'].items():
				name = key[7:]  # 移除'module.'前缀
				new_state_dict[name] = val
			# 加载调整后的模型状态
			self.model.load_state_dict(new_state_dict)
		# 恢复模型的迭代次数和训练轮数
		self.iters = checkpoint['iters']
		self.startEpoch = checkpoint['epoch']
		# 如果是恢复训练（而非微调），则加载优化器状态
		if self.params.resuming:
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加运行编号、YAML配置文件路径、配置名称等参数
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--epsilon_factor", default=0, type=float)

    # 解析命令行参数
    args = parser.parse_args()

    # 从YAML文件加载参数并更新epsilon_factor
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['epsilon_factor'] = args.epsilon_factor

    # 设置世界大小（用于分布式训练）
    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
        params['world_size'] = int(os.environ['WORLD_SIZE'])

    # 初始化本地和全局排名变量
    world_rank = 0
    local_rank = 0
    # 如果是分布式训练环境，初始化进程组
    if params['world_size'] > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = int(os.environ["LOCAL_RANK"])
        args.gpu = local_rank
        world_rank = dist.get_rank()
        # 更新批次大小为每个进程的批次大小
        params['global_batch_size'] = params.batch_size
        params['batch_size'] = int(params.batch_size // params['world_size'])

    # 设置当前使用的CUDA设备
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # 设置实验目录和检查点路径
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, 'training_checkpoints/'))

    # 更新参数字典中的实验目录和检查点路径
    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')

    # 检查是否从检查点恢复训练
    args.resuming = True if os.path.isfile(params.checkpoint_path) else False
    params['resuming'] = args.resuming
    params['local_rank'] = local_rank
    params['enable_amp'] = args.enable_amp

    # 设置wandb日志的名称和分组
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = "era5_precip" + args.config
    params['project'] = "ERA5_precip"
    params['entity'] = "flowgan"
    if world_rank == 0:
        # 设置日志文件和记录软件版本
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    # 根据运行排名设置wandb和屏幕日志
    params['log_to_wandb'] = (world_rank == 0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank == 0) and params['log_to_screen']

    # 设置输入和输出通道数
    params['in_channels'] = np.array(params['in_channels'])
    params['out_channels'] = np.array(params['out_channels'])
    if params.orography:
        params['N_in_channels'] = len(params['in_channels']) + 1
    else:
        params['N_in_channels'] = len(params['in_channels'])
    params['N_out_channels'] = len(params['out_channels'])

    # 如果是主节点，保存超参数到YAML文件
    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    # 初始化训练器并开始训练
    trainer = Trainer(params, world_rank)
    trainer.train()
    # 训练完成后记录日志
    logging.info('DONE ---- rank %d' % world_rank)

