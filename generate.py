import argparse
import os

import cv2
import numpy as np
import torch
import sys
from tqdm import tqdm

import utils.imgops as ops
import utils.architecture.architecture as arch

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='input', help='Input folder')
parser.add_argument('--output', default='output', help='Output folder')
parser.add_argument('--reverse', help='Reverse Order', action="store_true")
parser.add_argument('--tile_size', default=512,
                    help='Tile size for splitting', type=int)
parser.add_argument('--seamless', action='store_true',
                    help='Seamless upscaling')
parser.add_argument('--mirror', action='store_true',
                    help='Mirrored seamless upscaling')
parser.add_argument('--replicate', action='store_true',
                    help='Replicate edge pixels for padding')
parser.add_argument('--cpu', action='store_true',
                    help='Use CPU instead of CUDA')
parser.add_argument('--ishiiruka', action='store_true',
                    help='Save textures in the format used in Ishiiruka Dolphin material map texture packs')
parser.add_argument('--ishiiruka_texture_encoder', action='store_true',
                    help='Save textures in the format used by Ishiiruka Dolphin\'s Texture Encoder tool')
parser.add_argument('--no-normal', help='Do not generate normal maps', action="store_true")
parser.add_argument('--no-other', help='Do not generate roughness and displacement maps', action="store_true")
parser.add_argument('--skip-existing', help='Skip maps that were already generated', action="store_true")
args = parser.parse_args()

if not os.path.exists(args.input):
    print('Error: Folder [{:s}] does not exist.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.input):
    print('Error: Folder [{:s}] is a file.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.output):
    print('Error: Folder [{:s}] is a file.'.format(args.output))
    sys.exit(1)
elif not os.path.exists(args.output):
    os.mkdir(args.output)

device = torch.device('cpu' if args.cpu else 'cuda')

input_folder = os.path.normpath(args.input)
output_folder = os.path.normpath(args.output)

NORMAL_MAP_MODEL = 'utils/models/1x_NormalMapGenerator-CX-Lite_200000_G.pth'
OTHER_MAP_MODEL = 'utils/models/1x_FrankenMapGenerator-CX-Lite_215000_G.pth'

def process(img, model):
    if model is None:
        return None

    img = img * 1. / np.iinfo(img.dtype).max
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    output = model(img_LR).data.squeeze(
        0).float().cpu().clamp_(0, 1).numpy()
    output = output[[2, 1, 0], :, :]
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.).round()
    return output

def load_model(model_path):
    global device
    state_dict = torch.load(model_path)
    model = arch.RRDB_Net(3, 3, 32, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu',
                            mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)

images=[]
for root, _, files in os.walk(input_folder):
    for file in sorted(files, reverse=args.reverse):
        if file.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tga']:
            images.append(os.path.join(root, file))
models = []
if not args.no_normal:
    # NORMAL MAP
    models.append(load_model(NORMAL_MAP_MODEL))
if not args.no_other:
    # ROUGHNESS/DISPLACEMENT MAPS
    models.append(load_model(OTHER_MAP_MODEL))

if not models:
    sys.exit('at least one model should be used')
    
for idx, path in enumerate(tqdm(images), 1):
    base = os.path.splitext(os.path.relpath(path, input_folder))[0]
    output_dir = os.path.dirname(os.path.join(output_folder, base))
    os.makedirs(output_dir, exist_ok=True)
    print(idx, path)

    local_models = models[:]

    if not args.no_normal:
        normal_name = '{:s}.nrm.png'.format(base) if args.ishiiruka else '{:s}_Normal.png'.format(base)
        normal_file_path = os.path.join(output_folder, normal_name)
        if args.skip_existing and os.path.exists(normal_file_path):
            print('skipping already existing normal')
            local_models[0] = None

    if not args.no_other:
        idx = 0 if args.no_normal else 1

        rough_name = '{:s}.spec.png'.format(base) if args.ishiiruka else '{:s}_Roughness.png'.format(base)
        rough_file_path = os.path.join(output_folder, rough_name)

        displ_name = '{:s}.bump.png'.format(base) if args.ishiiruka else '{:s}_Displacement.png'.format(base)
        displ_file_path = os.path.join(output_folder, displ_name)

        if args.skip_existing and os.path.exists(rough_file_path) and os.path.exists(displ_file_path):
            print('skipping already existing other')
            local_models[idx] = None

    if not [m for m in local_models if m is not None]:
        continue

    # read image
    try:
        img = cv2.imread(path, cv2.cv2.IMREAD_COLOR)
    except:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
    # Seamless modes
    if args.seamless:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
    elif args.mirror:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
    elif args.replicate:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)

    img_height, img_width = img.shape[:2]

    # Whether or not to perform the split/merge action
    do_split = img_height > args.tile_size or img_width > args.tile_size

    if do_split:
        rlts = ops.esrgan_launcher_split_merge(img, process, models, scale_factor=1, tile_size=args.tile_size)
    else:
        rlts = [process(img, model) for model in models]

    if args.seamless or args.mirror or args.replicate:
        rlts = [ops.crop_seamless(rlt) for rlt in rlts]

    normal_map = None
    if not args.no_normal:
        normal_map = rlts[0]

    roughness = None
    displacement = None
    if not args.no_other:
        idx = 0 if args.no_normal else 1
        roughness = rlts[idx][:, :, 1]
        displacement = rlts[idx][:, :, 0]

    if args.ishiiruka_texture_encoder:
        r = 255 - roughness
        g = normal_map[:, :, 1]
        b = displacement
        a = normal_map[:, :, 2]
        output = cv2.merge((b, g, r, a))
        cv2.imwrite(os.path.join(output_folder, '{:s}.mat.png'.format(base)), output)
    else:
        if normal_map is not None:
            normal_name = '{:s}.nrm.png'.format(base) if args.ishiiruka else '{:s}_Normal.png'.format(base)
            normal_file_path = os.path.join(output_folder, normal_name)
            cv2.imwrite(normal_file_path, normal_map)

        if roughness is not None:
            rough_img = 255 - roughness if args.ishiiruka else roughness
            cv2.imwrite(rough_file_path, rough_img)

        if displacement is not None:
            cv2.imwrite(displ_file_path, displacement)
