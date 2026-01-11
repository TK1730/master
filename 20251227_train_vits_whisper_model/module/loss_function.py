#encoding:utf-8

import random
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models,transforms
import torchvision.utils as vutils
import torch.nn.init as init
from torch.autograd import Function
import torch.nn.functional as F

import torchaudio

def discriminator_adversarial_loss(discriminator_real_outputs, discriminator_fake_outputs):
	loss = 0
	real_losses = []
	fake_losses = []
	for dr, df in zip(discriminator_real_outputs, discriminator_fake_outputs):
		dr = dr.float()
		df = df.float()
		real_loss = torch.mean((1-dr)**2)
		fake_loss = torch.mean(df**2)
		loss += (real_loss + fake_loss)
		real_losses.append(real_loss.item())
		fake_losses.append(fake_loss.item())
	return loss, real_losses, fake_losses

def generator_adversarial_loss(discriminator_fake_outputs):
	loss = 0
	generator_losses = []
	for dg in discriminator_fake_outputs:
		dg = dg.float()
		l = torch.mean((1-dg)**2)
		generator_losses.append(l)
		loss += l
	return loss, generator_losses

def kl_divergence_loss(z_p, logs_q, m_p, logs_p, z_mask):
	""" 条件付きKLダイバージェンスを計算する関数

	Args:
		z_p (_type_): flowによってサンプリングされた潜在変数z
		logs_q (_type_): PosteriorEncoderによって計算された対数分散
		m_p (_type_): TextEncoderによって計算された平均
		logs_p (_type_): TextEncoderによって計算された対数分散
		z_mask (_type_): 潜在変数zのマスク

	Returns:
		_type_: klダイバージェンスloss
	"""
	z_p = z_p.float()
	logs_q = logs_q.float()
	m_p = m_p.float()
	logs_p = logs_p.float()
	z_mask = z_mask.float()

	kl = logs_p - logs_q - 0.5
	kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
	kl = torch.sum(kl * z_mask)
	l = kl / torch.sum(z_mask)
	return l

def feature_loss(feature_map_real, feature_map_fake):
	loss = 0
	for fmap_real, fmap_fake in zip(feature_map_real, feature_map_fake):
		for fmreal, fmfake in zip(fmap_real, fmap_fake):
			fmreal = fmreal.float().detach()
			fmfake = fmfake.float()
			loss += torch.mean(torch.abs(fmreal - fmfake))
	return loss * 2

def info_nce_loss(v_f, w_f, temperature=1.0):
    """ コントラスト学習でよく使われる損失関数
    v_f, w_f: [B, D] -> batch, embedding dimension
    temperature: 温度係数
    """
    # L2正規化
    v_f = F.normalize(v_f, p=2.0, dim=1)
    w_f = F.normalize(w_f, p=2.0, dim=1)
    # 類似度の計算
    logits = v_f @ w_f.T / temperature  # [B, B]
    labels = torch.arange(v_f.size(0), device=v_f.device)
    return F.cross_entropy(logits, labels)


if __name__ == "__main__":
    
    def kl_divergence_loss_test(z_p, logs_q, m_p, logs_p):
        z_p = z_p.float()
        logs_q = logs_q.float()
        m_p = m_p.float()
        logs_p = logs_p.float()
        kl = logs_p - logs_q - 0.5
        kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
        kl = torch.sum(kl)
        l = kl
        return l.item()

    # 固定値
    batch_size, dim = 2, 3
    z_p = torch.zeros(batch_size, dim)
    logs_q = torch.zeros(batch_size, dim)
    m_p = torch.zeros(batch_size, dim)

    # logs_pを-5から5まで変化させてKLロスを計算
    logs_p_range = np.linspace(-5, 5, 100)
    kl_values = []
    for lp in logs_p_range:
        logs_p = torch.full((batch_size, dim), lp)
        kl = kl_divergence_loss_test(z_p, logs_q, m_p, logs_p)
        kl_values.append(kl)
    
    fig =plt.figure(figsize=(10, 6))
    plt.plot(logs_p_range, kl_values)
    plt.xlabel("logs_p")
    plt.ylabel("KL Divergence Loss")
    plt.title("KL Divergence Loss vs logs_p")
    plt.grid()
    # plt.savefig("kl_divergence_loss.png")  # Save the plot as an image file
    plt.show()