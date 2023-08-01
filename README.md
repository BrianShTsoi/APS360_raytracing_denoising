# APS360_Raytracing_Denoising

## Instructions (Colab)
- Make sure you have access to the following folder on your Google Drive:
https://drive.google.com/drive/u/0/folders/1WaWD-KgrgYsRE5i0a-IxtwYJo2oypw5h
- Upload APS360Project.ipynb to Google Colab and run. Select yes to provide permission to access your Google Drive.

## Instructions (local/CPU)
- Download this folder: https://drive.google.com/drive/u/0/folders/1WaWD-KgrgYsRE5i0a-IxtwYJo2oypw5h, and save it alongside aps360project.py.
- Open aps360project.py and set TRY_USE_GPU to False.

## Instructions (local/GPU)
- Download this folder: https://drive.google.com/drive/u/0/folders/1WaWD-KgrgYsRE5i0a-IxtwYJo2oypw5h, and save it alongside aps360project.py.
- Install the CUDA 11.8 toolkit (https://developer.nvidia.com/cuda-11-8-0-download-archive)
- Run ```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```.
  If this fails, add Python3 directories to the path variable. You can do this by reinstalling Python3 and selecting "Add Python3 to PATH".
- If you're using PyCharm, select "Inherit global site-packages" when making a new project.
- To verify everything is installed correctly, run the following in a python console:
  ```
  import torch
  torch.cuda.is_available()
  ```
