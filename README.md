# PyTorch SDSR-OCT
----- A PyTorch implementation of ["Simultaneous denoising and super-resolution of optical coherence tomography images based on generative adversarial network"](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-27-9-12289)

## 1. the description of files in this repository:
* 1. `"dataset.py"` code for load training and testing data
* 2. `"main_2x.py main_4x.py main_8x.py"` code for training and testing of the models
* 3. `"models.py"` code for network architecture
* 4. `"vis_tools.py"` code for visualizing
* 5. `"metrics.py"` code for evaluating metrics and selecting ROIs

## 2. Citation
* If you use this code, please cite our work: 

```
@article{huang2019simultaneous,
  title={Simultaneous denoising and super-resolution of optical coherence tomography images based on generative adversarial network},
  author={Huang, Yongqiang and Lu, Zexin and Shao, Zhimin and Ran, Maosong and Zhou, Jiliu and Fang, Leyuan and Zhang, Yi},
  journal={Optics express},
  volume={27},
  number={9},
  pages={12289--12307},
  year={2019},
  publisher={Optical Society of America}
}
```
## 3. Contact
* Any questions about this code, please contact the author: tsmotlp@163.com
