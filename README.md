# SaliencyCut and AlphaMatting

## 1. Matting 方法

(1) SaliencyCut  
-----------

根据论文:  

[1] MM Cheng, NJ Mitra, X Huang, PHS Torr SM Hu. Global Contrast based Salient 
    Region Detection. IEEE CVPR, p. 409-416, 2011.    

中提出的 SaliencyCut 实现的算法, 其本质是对 GrabCut 算法做了改进, 去掉了手工标注, 直接使用 Saliency 图.  

因此是一种 Unsupervised salient object segmentation. 该算法原本是程明明教授团队实现的, 不过他们实现的是 Windows 版本, 我在这里做了移植, 将平台相关的代码用标准库(包括 boost 库) 替换, 使其具有跨平台性.   

(2) AlphaMatting  
---------

根据论文:   

[Shared Sampling for Real-Time Alpha Matting](http://inf.ufrgs.br/~eslgastal/SharedMatting/)  

  Eduardo S. L. Gastal and Manuel M. Oliveira  
  Computer Graphics Forum. Volume 29 (2010), Number 2.  
  Proceedings of Eurographics 2010, pp. 575-584.  

这个实现并不是作者给出的实现, 作者给出的只是一个二进制文件和一个动态库文件, 可以在这里下载:   http://inf.ufrgs.br/~eslgastal/SharedMatting/SharedMatting-v1.0-Linux-x64.zip.  

这个版本的实现中并没有使用 GPU 并行加速, 因此使用这个版本测出的运行时间可能和论文上的有出入.   

## 2. Saliency 方法

(1) FASA   
------------

FASA: Fast, Accurate, and Size-Aware Salient Object Detection  


(2) Global Contrast Based On Region Contrast  
------------

Global Contrast based Salient Region Detection. 
  MM Cheng, NJ Mitra, X Huang, PHS Torr SM Hu. 
  IEEE CVPR, p. 409-416, 2011. 

(3) GMR   
------------

  @inproceedings{yang2013saliency,  
  title={Saliency detection via graph-based manifold ranking},  
  author={Yang, Chuan and Zhang, Lihe and Lu, Huchuan, Ruan, Xiang and Yang, Ming-Hsuan},  
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2013 IEEE Conference on},  
  pages={3166--3173},  
  year={2013},  
  organization={IEEE}  
  }  


## 3. 编译

Requirement   
------------

* Boost version: 1.54.0   
* OpenCV 2410

编译   
---------

```bash
mkdir build && cd build
cmake ..
make 
```

## 4. Example  

```bash
./salmat /path/to/an/imageDir
```

详细使用方法可参考 main.cpp 中提供的 salmat_demo() 函数.   

