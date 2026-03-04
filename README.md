# BiEvLight (CVPR 2026)

Official PyTorch implementation for the " BiEvLight: Bi-level Learning of Task-Aware Event Refinement for Low-Light Image Enhancement" (CVPR 2026).

<p align="center">
    <b>BiEvLight (CVPR 2026)</b>:
    🌐 <a href="xxx" target="_blank">Project</a> | 📃 <a href="xxx" target="_blank">Paper</a> | 🖼️ <a href="xxx" target="_blank">Poster</a> <br>
</p>


**Authors**: [Zishu Yao](https://github.com/iijjlk/)<sup>[:email:️](mailto:zishuyao98@gmail.com)</sup>, Xiang-Xiang Su, Shengning Zhou, Guang-Yong Chen, Xing Chen, * Fu Zhou University*

**Feel free to ask questions. If our work helps, please don't hesitate to give us a :star:!**


## :rocket: News
<!-- - [ ] Provide a script for inference on the user's own video -->

- [x] 2024/03/05: Initialize the repository
- [x] 2026/02/21: :tada: :tada: Our paper was accepted in CVPR'2026

## :bookmark: Table of Content
1. [Code](#code)
2. [Citation](#citation)
3. [Contact](#contact)
4. [License and Acknowledgement](#license-and-acknowledgement)



## Code Will be public soon!



[//]: # (### Installation)

[//]: # (* Dependencies: [Miniconda]&#40;https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh&#41;, [CUDA Toolkit 11.1.1]&#40;https://developer.nvidia.com/cuda-11.1.1-download-archive&#41;, [torch 1.10.2+cu111]&#40;https://download.pytorch.org/whl/cu111/torch-1.10.2%2Bcu111-cp37-cp37m-linux_x86_64.whl&#41;, and [torchvision 0.11.3+cu111]&#40;https://download.pytorch.org/whl/cu111/torchvision-0.11.3%2Bcu111-cp37-cp37m-linux_x86_64.whl&#41;.)

[//]: # ()
[//]: # (* Run in Conda)

[//]: # ()
[//]: # (    ```bash)

[//]: # (    conda create -y -n evtexture python=3.7)

[//]: # (    conda activate evtexture)

[//]: # (    pip install torch-1.10.2+cu111-cp37-cp37m-linux_x86_64.whl)

[//]: # (    pip install torchvision-0.11.3+cu111-cp37-cp37m-linux_x86_64.whl)

[//]: # (    git clone https://github.com/DachunKai/EvTexture.git)

[//]: # (    cd EvTexture && pip install -r requirements.txt && python setup.py develop)

[//]: # (    ```)

[//]: # (* Run in Docker :clap:)

[//]: # ()
[//]: # (  Note: before running the Docker image, make sure to install nvidia-docker by following the [official instructions]&#40;https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html&#41;.)

[//]: # ()
[//]: # (  [Option 1] Directly pull the published Docker image we have provided from [Alibaba Cloud]&#40;https://cr.console.aliyun.com/cn-hangzhou/instances&#41;.)

[//]: # (  ```bash)

[//]: # (  docker pull registry.cn-hangzhou.aliyuncs.com/dachunkai/evtexture:latest)

[//]: # (  ```)

[//]: # ()
[//]: # (  [Option 2] We also provide a [Dockerfile]&#40;https://github.com/DachunKai/EvTexture/blob/main/docker/Dockerfile&#41; that you can use to build the image yourself.)

[//]: # (  ```bash)

[//]: # (  cd EvTexture && docker build -t evtexture ./docker)

[//]: # (  ```)

[//]: # (  The pulled or self-built Docker image containes a complete conda environment named `evtexture`. After running the image, you can mount your data and operate within this environment.)

[//]: # (  ```bash)

[//]: # (  source activate evtexture && cd EvTexture && python setup.py develop)

[//]: # (  ```)

[//]: # (### Test)

[//]: # (1. Download the pretrained models from &#40;[Releases]&#40;https://github.com/DachunKai/EvTexture/releases&#41; / [Onedrive]&#40;https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab&#41; / [Google Drive]&#40;https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing&#41; / [Baidu Cloud]&#40;https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg&#41;&#40;n8hg&#41;&#41; and place them to `experiments/pretrained_models/EvTexture/`. The network architecture code is in [evtexture_arch.py]&#40;https://github.com/DachunKai/EvTexture/blob/main/basicsr/archs/evtexture_arch.py&#41;.)

[//]: # (    * *EvTexture_REDS_BIx4.pth*: trained on REDS dataset with BI degradation for $4\times$ SR scale.)

[//]: # (    * *EvTexture_Vimeo90K_BIx4.pth*: trained on Vimeo-90K dataset with BI degradation for $4\times$ SR scale.)

[//]: # ()
[//]: # (2. Download the preprocessed test sets &#40;including events&#41; for REDS4 and Vid4 from &#40;[Releases]&#40;https://github.com/DachunKai/EvTexture/releases&#41; / [Onedrive]&#40;https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab&#41; / [Google Drive]&#40;https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing&#41; / [Baidu Cloud]&#40;https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg&#41;&#40;n8hg&#41;&#41;, and place them to `datasets/`.)

[//]: # (    * *Vid4_h5*: HDF5 files containing preprocessed test datasets for Vid4.)

[//]: # ()
[//]: # (    * *REDS4_h5*: HDF5 files containing preprocessed test datasets for REDS4.)

[//]: # ()
[//]: # (3. Run the following command:)

[//]: # (    * Test on Vid4 for 4x VSR:)

[//]: # (      ```bash)

[//]: # (      ./scripts/dist_test.sh [num_gpus] options/test/EvTexture/test_EvTexture_Vid4_BIx4.yml)

[//]: # (      ```)

[//]: # (    * Test on REDS4 for 4x VSR:)

[//]: # (      ```bash)

[//]: # (      ./scripts/dist_test.sh [num_gpus] options/test/EvTexture/test_EvTexture_REDS4_BIx4.yml)

[//]: # (      ```)

[//]: # (      This will generate the inference results in `results/`. The output results on REDS4 and Vid4 can be downloaded from &#40;[Releases]&#40;https://github.com/DachunKai/EvTexture/releases&#41; / [Onedrive]&#40;https://1drv.ms/f/c/2d90e71fb9eb254f/EnMm8c2mP_FPv6lwt1jy01YB6bQhoPQ25vtzAhycYisERw?e=DiI2Ab&#41; / [Google Drive]&#40;https://drive.google.com/drive/folders/1oqOAZbroYW-yfyzIbLYPMJ2ZQmaaCXKy?usp=sharing&#41; / [Baidu Cloud]&#40;https://pan.baidu.com/s/161bfWZGVH1UBCCka93ImqQ?pwd=n8hg&#41;&#40;n8hg&#41;&#41;.)

[//]: # ()
[//]: # (### Data Preparation)

[//]: # (* Both video and event data are required as input, as shown in the [snippet]&#40;https://github.com/DachunKai/EvTexture/blob/main/basicsr/archs/evtexture_arch.py#L70&#41;. We package each video and its event data into an [HDF5]&#40;https://docs.h5py.org/en/stable/quick.html#quick&#41; file.)

[//]: # ()
[//]: # (* Example: The structure of `calendar.h5` file from the Vid4 dataset is shown below.)

[//]: # ()
[//]: # (  ```arduino)

[//]: # (  calendar.h5)

[//]: # (  ├── images)

[//]: # (  │   ├── 000000 # frame, ndarray, [H, W, C])

[//]: # (  │   ├── ...)

[//]: # (  ├── voxels_f)

[//]: # (  │   ├── 000000 # forward event voxel, ndarray, [Bins, H, W])

[//]: # (  │   ├── ...)

[//]: # (  ├── voxels_b)

[//]: # (  │   ├── 000000 # backward event voxel, ndarray, [Bins, H, W])

[//]: # (  │   ├── ...)

[//]: # (  ```)

[//]: # (* To simulate and generate the event voxels, refer to the dataset preparation details in [DataPreparation.md]&#40;https://github.com/DachunKai/EvTexture/blob/main/datasets/DataPreparation.md&#41;.)

[//]: # ()
[//]: # (### Inference on your own video)

[//]: # (:hammer_and_wrench: We are developing a convenient script to allow users to quickly use our EvTexture model to upscale their own videos. However, our spare time is limited, so please stay tuned!)

[//]: # ()
[//]: # (## :blush: Citation)

[//]: # (If the code and pre-trained models facilitate your research, please consider citing the corresponding papers. :smiley:)

[//]: # (```)

[//]: # (@article{kai2026evtexture++,)

[//]: # (  title={{E}v{T}exture++: {E}vent-{D}riven {T}exture {E}nhancement for {V}ideo {S}uper-{R}esolution},)

[//]: # (  author={Kai, Dachun and Lu, Jiayao and Zhang, Yueyi and Sun, Xiaoyan},)

[//]: # (  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},)

[//]: # (  year={2026},)

[//]: # (  doi={10.1109/TPAMI.2026.3660020})

[//]: # (})

[//]: # ()
[//]: # (@inproceedings{kai2024evtexture,)

[//]: # (  title={{E}v{T}exture: {E}vent-driven {T}exture {E}nhancement for {V}ideo {S}uper-{R}esolution},)

[//]: # (  author={Kai, Dachun and Lu, Jiayao and Zhang, Yueyi and Sun, Xiaoyan},)

[//]: # (  booktitle={Proceedings of the 41st International Conference on Machine Learning},)

[//]: # (  pages={22817--22839},)

[//]: # (  year={2024},)

[//]: # (  volume={235},)

[//]: # (  publisher={PMLR})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (## Contact)

[//]: # (If you meet any problems, please describe them in issues or contact:)

[//]: # (* Dachun Kai: <dachunkai@mail.ustc.edu.cn>)

[//]: # ()
[//]: # (## License and Acknowledgement)

[//]: # (This project is released under the Apache-2.0 license. Our work is built upon [BasicSR]&#40;https://github.com/XPixelGroup/BasicSR&#41;, which is an open source toolbox for image/video restoration tasks. Thanks to the inspirations and codes from [RAFT]&#40;https://github.com/princeton-vl/RAFT&#41;, [event_utils]&#40;https://github.com/TimoStoff/event_utils&#41; and [EvTexture-jupyter]&#40;https://github.com/camenduru/EvTexture-jupyter&#41;.)
 c9b2160363a226f0d5ac27208f934bff8f191940
