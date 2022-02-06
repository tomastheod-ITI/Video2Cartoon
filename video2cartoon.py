
# -------------------------------------------------------------------------------------------------
#  Import Packages
# -------------------------------------------------------------------------------------------------

import torch
from model.Transformer import Transformer

import numpy as np
import cv2

import os
import argparse
import requests
import math
from datetime import datetime


# %%

# -------------------------------------------------------------------------------------------------
#  Parse Arguments
# -------------------------------------------------------------------------------------------------

arg_parser = argparse.ArgumentParser(description = 'Process the input file (image or video) and ' +
                                                   'create a cartoon version of it.')

arg_parser.add_argument('file',            help = 'The input file to process (image or video).')

arg_parser.add_argument('--output_dir',    default = 'output',
                                           help = 'The output directory to store the results. Default \
                                                   is output.')

arg_parser.add_argument('--style',         default = 'shinkai',
                                           help = 'The cartoon style to use for creating the output. \
                                                   Available styles: hayao, shinkai, hosoda, paprika. \
                                                   The first time you run a specific style, the \
                                                   corresponding model parameters will be downloaded \
                                                   automatically and placed inside the model_weights \
                                                   directory, which may take a few seconds. Default \
                                                   is shinkai.')

arg_parser.add_argument('--reduce_ratio',  default = 1, type = float,
                                           help = 'Resize the input frames by reduce_ratio in order \
                                                   to make them smaller. This is useful if your \
                                                   system runs out of memory when processing the input. \
                                                   reduce_ratio should be a number in (0, 1]. \
                                                   Default is 1 (no reduction).')

arg_parser.add_argument('--batch_size',    default = 1, type = int,
                                           help = 'The number of frames to process at once. Depending \
                                                   on your system, batch sizes > 1 may provide a \
                                                   significant speed improvement for lower resolution \
                                                   videos. Default is 1.')

arg_parser.add_argument('--image_format',  default = 'jpg',
                                           help = 'The format to use when creating images. Available \
                                                   formats are jpg and png. Default is jpg.')

arg_parser.add_argument('--video_codec',   default = 'DIVX',
                                           help = 'The FourCC identifier of the codec to use when \
                                                   creating videos. The codec must be installed in \
                                                   your system. See https://www.fourcc.org/ for more \
                                                   details. Default is DIVX.')

# use temp_args in arg_parser if you want to run the code line-by-line
#temp_args = ['media/lake.jpg', '--style', 'paprika']

args = arg_parser.parse_args()


del arg_parser


# %%

# -------------------------------------------------------------------------------------------------
#  Check Arguments
# -------------------------------------------------------------------------------------------------

# check that the input file exists
assert os.path.isfile(args.file), f'Cannot find input file {args.file}'

# check output directory
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# check weights directory
if not os.path.exists('model_weights'):
    os.mkdir('model_weights')

# check style
valid_styles = {'hayao', 'shinkai', 'hosoda', 'paprika'}
input_style = args.style.lower()
assert input_style in valid_styles, f'style must be one of: {", ".join(valid_styles)}, not {input_style}'

style_name = f'{input_style[0].upper()}{input_style[1:]}'
weights_name = f'{style_name}_net_G_float.pth'
weights_file = os.path.join('model_weights', weights_name)
if not os.path.isfile(weights_file):
    # download model weights
    weights_url = f'http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/{weights_name}'
    print(weights_url)
    print(f'Downloading model weights for style: {style_name}...')

    req = requests.get(weights_url, allow_redirects = False, timeout = 60)
    print(f'{req.status_code} -> {req.reason}')

    if req.status_code == 200:
        with open(weights_file, mode = 'wb') as f:
            f.write(req.content)
    else:
        raise RuntimeError('Could not download model weights. You can download them manually from ' +
                          f'{weights_url} and place them in the model_weights directory.')

# check input file can be opened
vc = cv2.VideoCapture(args.file)
assert vc.isOpened(), f'{args.file} could not be opened.'

# check reduce_ratio
assert 0 < args.reduce_ratio <= 1, f'reduce_ratio must be > 0 and <= 1, not {args.reduce_ratio}'

img_scaling = False if args.reduce_ratio == 1 else True

img_w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
img_h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))

min_dim = 4
assert img_w > min_dim and img_h > min_dim, f'Image dimensions must be > {min_dim}, not {img_w} x {img_h}.'

if img_scaling:
    img_w = math.floor(args.reduce_ratio * img_w)
    img_h = math.floor(args.reduce_ratio * img_h)
    assert img_w > min_dim and img_h > min_dim, f'Image dimensions after scaling must be > {min_dim}, not ' + \
                                                f'{img_w} x {img_h}. reduce_ratio is probably set too low.'

# check batch_size
assert isinstance(args.batch_size, int) and args.batch_size >= 1, 'batch_size must be an integer >= 1.'

# calculate output file name
input_name_ext = os.path.splitext(os.path.split(args.file)[-1])
output_name = f'{input_name_ext[0]}_{style_name}'
output_file = os.path.join(args.output_dir, output_name)

# check image_format and video_codec
input_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
if input_frames > 1:
    # create video
    fps    = vc.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*args.video_codec)

    vw = cv2.VideoWriter(f'{output_file}.mkv', fourcc, fps, (img_w, img_h))
    assert vw.isOpened(), f'{output_file}.mkv could not be created.'
else:
    # create image
    valid_formats = {'jpg', 'png'}
    image_format = args.image_format.lower()
    if image_format == 'jpeg':
        image_format = 'jpg'

    assert image_format in valid_formats, f'image_format must be one of {", ".join(valid_formats)}, ' + \
                                          f'not {image_format}'

print(f'Input file: {"".join(input_name_ext)}, style: {style_name}, frames: {input_frames}, ' +
      f'output dimensions: {img_w} x {img_h}')


del valid_styles, input_style, weights_name, input_name_ext, style_name, output_name


# %%

# -------------------------------------------------------------------------------------------------
#  Load model
# -------------------------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer().to(device)

model.load_state_dict(torch.load(weights_file))
model.eval();


del weights_file


# %%

# -------------------------------------------------------------------------------------------------
#  Process input
# -------------------------------------------------------------------------------------------------

def process_image(img):
    # reduce image size if needed
    if img_scaling:
        img = cv2.resize(img, (img_w, img_h), interpolation = cv2.INTER_AREA)
    # change data type and scale to [0, 1]
    img = img.astype(np.float32) / 255
    # scale to [-1, 1]
    img = 2 * img - 1
    # bring channels to the front (h, w, c) -> (c, h, w)
    img = np.moveaxis(img, [0, 1, 2], [1, 2, 0])

    return img

def unprocess_image(img):
    # bring channels to the back (c, h, w) -> (h, w, c)
    img = np.moveaxis(img, [0, 1, 2], [2, 0, 1])
    # scale to [0, 1]
    img = (img + 1) / 2
    # change data type and scale to [0, 255]
    img = (img * 255).astype(np.uint8)

    return img


t_start = datetime.utcnow()
t1      = datetime.utcnow()

flag, img = vc.read()
assert flag, 'Cannot read input.'

batch_imgs = []
while flag:

    batch_imgs.append(img)
    while len(batch_imgs) < args.batch_size:
        flag, img = vc.read()
        if flag:
            batch_imgs.append(img)
        else:
            break

    res = [process_image(x) for x in batch_imgs]
    res = np.stack(res, axis=0)

    img_in = torch.tensor(res, dtype=torch.float32, device=device)

    # cartoonify images!
    with torch.no_grad():
        img_out = model(img_in)

    res = img_out.cpu().numpy()
    res = [unprocess_image(x) for x in res]

    if input_frames > 1:
        # write video
        for img in res:
            vw.write(img)
    else:
        # write image
        cv2.imwrite(f'{output_file}.{image_format}', res[0])

    del flag, img, res, img_in, img_out
    batch_imgs.clear()

    t2 = datetime.utcnow()

    dt = (t2 - t1).total_seconds()
    if dt > 10 and input_frames > 1:
        cur_pos = int(vc.get(cv2.CAP_PROP_POS_FRAMES))
        s = f'Processing frames... {(100 * cur_pos / input_frames):.1f}%'
        n_blank = 50 - len(s)
        print('\r', s, ' ' * n_blank, sep='', end='')
        t1 = datetime.utcnow()

    flag, img = vc.read()

vc.release()

if input_frames > 1:
    vw.release()

t_end = datetime.utcnow()

print()
print(f'Finished processing in {(t_end - t_start).total_seconds():.1f} sec')


