import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


#encoders = ['vits', 'vitb', 'vitl']
#number of variables increase from vits to vitl
encoder = 'vits'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE)

total_params = sum(param.numel() for param in depth_anything.parameters())
print('Total parameters: {:.2f}M'.format(total_params / 1e6))

depth_anything.eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'x264' might also be available
out_video = cv2.VideoWriter("output_video.mp4", fourcc, 30.0, (640,480))


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, raw_image = cap.read()

    if not ret:
        break

    #raw_image = cv2.resize(raw_image, (2560, 1600))

    #converting the raw image to RGB and then normalising the image pixel values for image processing
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    #getting the height and width of the raw image
    h, w = image.shape[:2]
    
    #transforming the image 
    image = transform({'image': image})['image']
    
    #converts the numpy array to a pytorch tensor(from_numpy), adds a batch dimension for deep learning(unsqueeze),
    #move tensor to specified device(.to())
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    #temporarily disable gradient computation
    with torch.no_grad():
        #running the image through the depth anything algorithm
        depth = depth_anything(image)
    
    #depth[None] adds batch dimension(like unsqueeze(0)), interpolate resizes the tensor, bilinear is just an algo for resizing
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    #.cpu() moves tensor to the cpu, .numpy() converts it to a numpy array(only valid for cpu tensors),
    #.astype(np.uint8) converts numpy array to uint8(common data type for image data)
    depth = depth.cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    # Write the frame to the video file
    out_video.write(depth_color)
    cv2.imshow('Depth Anything', depth_color)

    # Press q on keyboard to exit
    if (cv2.waitKey(1) & 0xFF) == ord('q') :
        break


    
cap.release()
out_video.release()
cv2.destroyAllWindows()