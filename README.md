# End-to-End Mulsemedia system


# Requirements
We attached a requirements.txt with all the neccessary requirements. However, following the steps of installation will install everything properly as well. We rather recommend that way.

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

# Usage

TODO

# Ref

The code is based on the library of [SLOWFAST](https://github.com/facebookresearch/SlowFast), maintaned by [Haoqi Fan](https://haoqifan.github.io/), [Yanghao Li](https://lyttonhao.github.io/), [Bo Xiong](https://www.cs.utexas.edu/~bxiong/), [Wan-Yen Lo](https://www.linkedin.com/in/wanyenlo/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/).