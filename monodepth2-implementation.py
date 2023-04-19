from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

def Initialize(model_name):
    assert model_name is not None, "No Model Provided"

    #if torch.cuda.is_available():
    #    device = torch.device("cuda")
    #else:
    device = torch.device("cpu")

    model_path = os.path.join("models", model_name)
    download_model_if_doesnt_exist(model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)    
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']

    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)
    depth_decoder.to(device)
    depth_decoder.eval()

    return encoder, depth_decoder, device, feed_width, feed_height

def ExtractDepthTensor(unsqueezed_tensor_frame, device, encoder, depth_decoder):
    unsqueezed_tensor_frame = unsqueezed_tensor_frame.to(device)
    return (depth_decoder(encoder(unsqueezed_tensor_frame)))[("disp", 0)]

def ExtractDepthTensorFromFrame(frame, device, encoder, depth_decoder):
    unsqueezed_tensor_frame = transforms.ToTensor()(frame).unsqueeze(0)
    return ExtractDepthTensor(unsqueezed_tensor_frame, device, encoder, depth_decoder).squeeze().cpu()

def GetValAtCoordFromTensor(at_x, at_y, tensor):
    #print(type(tensor))
    #print(tensor.size())
    return 100.0-((tensor[at_y,at_x]).item()*100)

def GetOutput(frame, device, encoder, depth_decoder):
    unsqueezed_tensor_frame = transforms.ToTensor()(frame).unsqueeze(0)
    disp = ExtractDepthTensor(unsqueezed_tensor_frame, device, encoder, depth_decoder).squeeze().cpu().numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    frame_out = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)

    display = np.zeros((frame_out.shape[0]*2, frame_out.shape[1], frame_out.shape[2]), np.uint8)
    display[:frame.shape[0], :frame.shape[1]] = frame
    display[frame.shape[0]:, :frame.shape[1]] = frame_out

    return display