import os
import argparse
import cv2
import torch
import numpy as np
from torch.cuda import amp
from tqdm import tqdm
from train import AnimeSegmentation, net_names

def resize_and_pad_image(input_img, target_size=1024):
    h, w = input_img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 创建一个 1024x1024 的黑色画布，将调整大小后的图像放置在中心
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
    parser.add_argument('--ckpt', type=str, default='/mnt/chenjh/Animatabler/anime-segmentation/isnetis.ckpt',
                        help='model checkpoint path')
    parser.add_argument('--input_dir', type=str, required=False,
                        help='/mnt/chenjh/Animatabler/input/sd-xl-Tpose')
    parser.add_argument('--output_dir', type=str, required=False,
                        help='/mnt/chenjh/Animatabler/input/sd-xl-Tpose-seg')
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

    # 创建输出目录
    os.makedirs(opt.output_dir, exist_ok=True)

    # 遍历输入文件夹中的所有图片文件
    for img_name in tqdm(os.listdir(opt.input_dir)):
        img_path = os.path.join(opt.input_dir, img_name)
        if not os.path.isfile(img_path):
            continue  # 跳过非文件项

        # 读取并调整图像
        img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            img_padded, x_offset, y_offset, new_w, new_h = resize_and_pad_image(img, target_size=1024)
        else:
            img_padded = img
            x_offset, y_offset, new_w, new_h = 0, 0, img.shape[1], img.shape[0]
        
        # 获取分割 mask
        mask = get_mask(model, img_padded, use_amp=not opt.fp32, s=opt.img_size)

        # 生成 RGBA 图像
        rgba_img = np.dstack((img_padded, (mask * 255).astype(np.uint8)))  # 使用 mask 作为 alpha 通道
        rgba_img = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGRA)

        # 保存分割后的 RGBA 图像为 PNG
        output_path = os.path.join(opt.output_dir, f"{os.path.splitext(img_name)[0]}.png")
        cv2.imwrite(output_path, rgba_img)

# CUDA_VISIBLE_DEVICES=4 python anime-segmentation/inference.py --input_dir /mnt/chenjh/Animatabler/input/anime-xl-Tpose --output_dir /mnt/chenjh/Animatabler/input/anime-xl-Tpose-seg
# CUDA_VISIBLE_DEVICES=0 python anime-segmentation/inference.py --input_dir /mnt/chenjh/Animatabler/text2anime/outputs --output_dir /mnt/chenjh/Animatabler/text2anime/outputs-animeseg