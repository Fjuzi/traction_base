# End-to-End Mulsemedia system


# Requirements
We attached a [requirements](https://github.com/Fjuzi/traction_base/blob/main/requirements.txt) list with all the neccessary requirements. However, following these steps for installation will setup everything with recommended settings.

# Install
Instalation of the scent and haptic predictor of the end-to-end mulsemedia system. 

Install CUDA.

Pytorch 1.8.0 is tested, with cuda = 11.0
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Fvcore to access SLOWFAST models
```
pip install -U 'git+https://github.com/facebookresearch/fvcore'
```

We cloned and included detectron2, but it still need to be installed
```
pip install -e detectron2_repo
```

Install slowfast project
```
python setup.py build develop
```

```
pip install pytorchvideo
```

Install CLIP (Optional) 
In case you want to use CLIP, it needs to be additionally installed from their [GitHub repository](https://github.com/openai/CLIP). 

```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

# Dataset

The instructions of obtaining the dataset can be found in [DATA.md](https://github.com/Fjuzi/traction_base/blob/main/data/DATA.md)

# Models
SlowFast and Scene recognition (DenseNet or ResNet) 

# Usage

Prediction with the parallell network:
```
python tools/run_net.py \
  --cfg configs/SLOWFAST_64x2_R101_50_50_kinetics.yaml \
  DATA.PATH_TO_DATA_DIR /data/Peter/Code/slowfast/data/ \
  TEST.CHECKPOINT_FILE_PATH /data/Peter/Code/slowfast/checkpoints/SLOWFAST_64x2_R101_50_50.pkl \
  TRAIN.ENABLE False
```

Prediction with CLIP zero-shot:
```
python ./custom_tools/test_clip.py
```


# Ref

The code is based on the library of [SLOWFAST](https://github.com/facebookresearch/SlowFast), maintaned by [Haoqi Fan](https://haoqifan.github.io/), [Yanghao Li](https://lyttonhao.github.io/), [Bo Xiong](https://www.cs.utexas.edu/~bxiong/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/).
