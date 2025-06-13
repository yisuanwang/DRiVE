import os
import argparse
import cv2
import torch
import numpy as np
from torch.cuda import amp
from train import AnimeSegmentation, net_names

def resize_and_pad_image(input_img, target_size=1024):
    h, w = input_img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Create a 1024x1024 black canvas and place the resized image in the center
    padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    
    return padded_img, x_offset, y_offset, new_w, new_h

def get_mask(model, input_img, use_amp=True, s=1024):
    input_img = (input_img / 255).astype(np.float32)
    img_input = np.transpose(input_img, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmpImg = torch.from_numpy(img_input).type(torch.FloatTensor).to(model.device)
    
    with torch.no_grad():
        if use_amp:
            with amp.autocast():
                pred = model(tmpImg)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmpImg)
        pred = pred.cpu().numpy()[0]
        pred = np.transpose(pred, (1, 2, 0))
        return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='isnet_is',
                        choices=net_names,
                        help='net name')
    parser.add_argument('--ckpt', type=str, default='anime-segmentation/isnetis.ckpt',
                        help='model checkpoint path')
    parser.add_argument('--input', type=str, required=True,
                        help='input image path')
    parser.add_argument('--output', type=str, required=True,
                        help='output image path')
    parser.add_argument('--img-size', type=int, default=1024,
                        help='input image size of the net')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='cpu or cuda:0')
    parser.add_argument('--fp32', action='store_true', default=False,
                        help='disable mixed precision')

    opt = parser.parse_args()
    print(opt)

    device = torch.device(opt.device)
    model = AnimeSegmentation.try_load(opt.net, opt.ckpt, opt.device, img_size=opt.img_size)
    model.eval()
    model.to(device)

    # 读取和调整图像
    img = cv2.cvtColor(cv2.imread(opt.input, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if img.shape[0] != 1024 or img.shape[1] != 1024:
        img_padded, x_offset, y_offset, new_w, new_h = resize_and_pad_image(img, target_size=1024)
    else:
        img_padded = img
        x_offset, y_offset, new_w, new_h = 0, 0, img.shape[1], img.shape[0]
    
    # 分割图像
    mask = get_mask(model, img_padded, use_amp=not opt.fp32, s=opt.img_size)

    # 生成 RGBA 图像
    rgba_img = np.dstack((img_padded, (mask * 255).astype(np.uint8)))  # 使用 mask 作为 alpha 通道
    rgba_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGRA)
    
    # 保存分割后的 RGBA 图像为 PNG
    cv2.imwrite(opt.output, rgba_img)


# python script_name.py --input $input_image_path --output $workspace/Anime.png
# python anime-segmentation/inferimg.py --input $input_image_path --output $workspace/Anime.png
# # python anime-segmentation/inferimg.py --input /mnt/chenjh/Animatabler/output/AnimeTpose-1108-01/00097-575592539/Anime.png --output /mnt/chenjh/Animatabler/output/AnimeTpose-1108-01/00097-575592539/lgm-full/1-rmbg_processed.png