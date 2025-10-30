import torch
from torch import nn
import logging
import torch.nn.functional as F
import os
import numpy as np
import cv2 as cv
import random

from models.load_spec_model import FMM_model
from load_data import build_loader_for_FMM


class FMM_Pretrain(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_modality = len(cfg.data.modalities_name)
        self.model_spe_list = [FMM_model(cfg.model.in_channels, cfg.model.down_num) for i in range(self.num_modality)]
        self.model_map_list = [FMM_model(self.num_modality - 1, cfg.model.down_num) for i in range(self.num_modality)]

    def pretrain(self, specific_encoder=True):
        # pretrain specific and mapping encoder
        for index_model in range(self.num_modality):
            name_of_modality = self.cfg.data.modalities_name[index_model].split('.')[0]
       
            if specific_encoder:
                model = self.model_spe_list[index_model].to(self.device)
                optim = torch.optim.AdamW(model.parameters(), lr=self.cfg.optim.lr,
                                          weight_decay=self.cfg.optim.weight_decay)
                self.train_spe_loop(model, optim, index_model, name_of_modality)
            else:
                model = self.model_map_list[index_model].to(self.device)
                optim = torch.optim.AdamW(model.encoder.parameters(), lr=self.cfg.optim.lr,
                                          weight_decay=self.cfg.optim.weight_decay)
                self.train_mapping_loop(model, optim, index_model, name_of_modality)

    def train_spe_loop(self, model, optim, index_model, name_of_modality):
        # pretrain specific encoder
        logging.info('Training specific encoder and decoder for {}'.format(name_of_modality))
        data_loader = self.build_dataloader()
        
        for epoch in range(1, self.cfg.train.epochs + 1):
            

            
            for iter, (inputs_vol_stack) in enumerate(data_loader):
                # 1 M Z H W -> M Z 1 H W
                inputs_vol_stack = inputs_vol_stack.permute(1, 2, 0, 3, 4).to(self.device)     
                inputs_vol = inputs_vol_stack[index_model]

                loss_for_vol = 0.0
                index_slice = 0
                acc_chunk = 0
                minibatch = self.cfg.train.mini_batch_size 
                while index_slice < len(inputs_vol):
                    acc_chunk += 1
                    start = index_slice
                    stop = index_slice + minibatch
                    index_slice += minibatch
                    if stop >= len(inputs_vol):
                        stop = len(inputs_vol)

                    inputs = inputs_vol[start:stop]
                    output = model(inputs)

                    loss = F.l1_loss(output, inputs, reduction='none').mean()
                    loss_for_vol += loss.item()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                logging.info('epoch: {}, vol index: {}, loss: {}'.format(epoch, iter, loss_for_vol / acc_chunk))

            if epoch % self.cfg.train.snapshot_freq_epoch == 0:
                self.save_ckp(model.encoder, model.decoder, name_of_modality, epoch, is_specific_encoder=True)

    def train_mapping_loop(self, model, optim, index_model, name_of_modality):
        # pretrain mapping encoder
        logging.info('Training mapping encoder for {}'.format(name_of_modality))
        data_loader = self.build_dataloader()

        model = self.load_decoder_parameter(model,name_of_modality)
        logging.info('loadding decoder for {}'.format(name_of_modality))

        for epoch in range(1, self.cfg.train.epochs + 1):

            
            for iter, (inputs_vol_stack) in enumerate(data_loader):
                # 1 M Z H W  -> M Z 1 H W
                inputs_vol_stack = inputs_vol_stack.permute(1, 2, 0, 3, 4).to(self.device)

                # Z 1 H W
                label_vol = inputs_vol_stack[index_model]
                
                # M-1 Z 1 H W -> M-1 Z H W -> Z M-1 1 H W 
                input_modalities = torch.cat((
                    inputs_vol_stack[:index_model],
                    inputs_vol_stack[index_model+1:]
                ), dim=0).squeeze(2).permute(1,0,2,3)


                input_modalities = self.random_zero_along_dim(input_modalities)
                
                loss_for_vol = 0.0
                index_slice = 0
                acc_chunk = 0
                minibatch = self.cfg.train.mini_batch_size
                while index_slice < len(label_vol):
                    acc_chunk += 1
                    start = index_slice
                    stop = index_slice + minibatch
                    index_slice += minibatch
                    if stop >= len(label_vol):
                        stop = len(label_vol)

                    inputs = input_modalities[start:stop]
                    label = label_vol[start:stop]
                    output = model(inputs)

                    loss = F.l1_loss(output, label, reduction='none').mean()
                    loss_for_vol += loss.item()

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                logging.info('epoch: {}, vol index: {}, loss: {}'.format(epoch, iter, loss_for_vol / acc_chunk))

            if epoch % self.cfg.train.snapshot_freq_epoch == 0:
                self.save_ckp(model.encoder, model.decoder, name_of_modality, epoch, is_specific_encoder=False)

    def load_decoder_parameter(self,model,name_of_modality):
        # loading specific decoder
        decoder_path = os.path.join(self.cfg.train.ckp_point_path, name_of_modality, "decoder",
                     f"model_epoch{self.cfg.train.decoder_ckp_for_load}.pth")
        ckp_decoder = torch.load(decoder_path, map_location='cpu')

        model.decoder.load_state_dict(ckp_decoder['model_state_dict'])
        for param in model.decoder.parameters():
            param.requires_grad = False

        return model

    def build_dataloader(self):
        return build_loader_for_FMM(self.cfg.data.data_store_path, self.cfg.data.modalities_name)

    def save_ckp(self, encoder, decoder, name_modality, epoch, is_specific_encoder=True):
        if is_specific_encoder:
            # specific
            save_dir_encoder = os.path.join(self.cfg.train.ckp_point_path, name_modality, "specific_encoder")
            save_dir_decoder = os.path.join(self.cfg.train.ckp_point_path, name_modality, "decoder")
            os.makedirs(save_dir_encoder, exist_ok=True)
            os.makedirs(save_dir_decoder, exist_ok=True)
            save_path_encoder = os.path.join(save_dir_encoder, f"model_epoch{epoch}.pth")
            save_path_decoder = os.path.join(save_dir_decoder, f"model_epoch{epoch}.pth")

            ckp_encoder = {
                'epoch': epoch,
                'model_state_dict': encoder.state_dict()
            }

            ckp_decoder = {
                'epoch': epoch,
                'model_state_dict': decoder.state_dict()
            }

            torch.save(ckp_encoder, save_path_encoder)
            torch.save(ckp_decoder, save_path_decoder)
            logging.info('saving specific encoder and decoder for epoch: {}'.format(epoch))
        else:
            # mapping encoder
            save_dir_encoder = os.path.join(self.cfg.train.ckp_point_path, name_modality, "mapping_encoder")
            os.makedirs(save_dir_encoder, exist_ok=True)
            save_path_encoder = os.path.join(save_dir_encoder, f"model_epoch{epoch}.pth")
  
            ckp_encoder = {
                'epoch': epoch,
                'model_state_dict': encoder.state_dict()
            }
            torch.save(ckp_encoder, save_path_encoder)
            logging.info('saving mapping encoder for epoch: {}'.format(epoch))
            
    def random_zero_along_dim(self, tensor, dim=1):

        n = tensor.shape[dim]
    
        keep_idx = random.randint(0, n - 1)
    
        mask = torch.ones(n, dtype=tensor.dtype, device=tensor.device)
        for i in range(n):
            if i != keep_idx and random.random() < 0.5: 
                mask[i] = 0
    
        mask[keep_idx] = 1

        shape = [1] * tensor.ndim
        shape[dim] = n
        mask = mask.view(*shape)
    
        return tensor * mask 
