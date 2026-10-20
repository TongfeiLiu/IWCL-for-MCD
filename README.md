# IWCL: Interactive Wavelet Commonality Learning for Unsupervised End-to-End Heterogeneous Change Detection

![Paper](https://img.shields.io/badge/Paper-TGRS%20P&RS-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)

This repository provides the official implementation of the paper:

Jianjian Xu, Tao Lei, Tongfei Liu*, Yingbo Wang, Xiaogang Du, Zhiyong Lv, and Maoguo Gong[J]. [IWCL: Interactive Wavelet Commonality Learning for Unsupervised End-to-End Heterogeneous Change Detection](https://ieeexplore.ieee.org/abstract/document/11455548),
**IEEE Transactions on Geoscience and Remote Sensing, 2026**, 64, Art no. 5614914.

---

## 🚀 Features

- ✅ Unsupervised multimodal change detection  
- ✅ Interactive wavelet commonality learning  
- ✅ Support End-to-End multimodal change detection  

---

## 📖 Abstract

Heterogeneous remote sensing change detection (CD) is a research hotspot in the field of remote sensing. However, due to the differences between different modalities, heterogeneous remote sensing images (HRSIs) are often difficult to compare and analyze directly. To address this challenge, this study hypothesizes that unchanged regions between HRSI exhibit significantly more commonality features in the wavelet domain than changed regions. Based on this, this article proposes an interactive wavelet commonality learning (IWCL) method for unsupervised end-to-end heterogeneous CD, which captures the wavelet commonality features between HRSI and identifies changes through their similarity measurement. The proposed IWCL consists of a Siamese wavelet encoder and two independent inverse wavelet decoders. Specifically, the wavelet encoder first decomposes the HRSI into multidirectional wavelet subbands, making the commonality features of the unchanged areas more prominent in multiple directions. Subsequently, we designed a cross-modal interactive learning (CMIL) module in the wavelet encoder to excavate deeper into these multidirectional wavelet commonality features. In addition, the wavelet commonality features extracted by the encoder are used to reconstruct the original image through two inverse wavelet decoders, which achieves unsupervised learning. Moreover, a multiscale commonality loss function is introduced to effectively guide the network to furth er improve the representation ability of wavelet commonality features at different scales. Finally, the difference image (DI) is acquired by measuring the similarity of wavelet commonality features between HRSI, and the binary change image is obtained by Otsu. Experiments on four heterogeneous CD datasets verify the effectiveness and superiority of the proposed IWCL in resolving heterogeneous CD. The code will be available at https://github.com/TongfeiLiu/IWCL-for-MCD.

The framework of the proposed CGSL is presented as follows:
![Framework of our proposed CGSL](https://github.com/TongfeiLiu/CGSL-for-MCD/blob/main/Figures/Freamwork%20of%20CGSL.png)

---

## 📊 Results Preview
**Change Maps**
![Change maps of the proposed IWCL](https://github.com/TongfeiLiu/CGSL-for-MCD/blob/main/Figures/BCIs.png)

## 📦 Installation

### Requirements
- Python 3.8+  
- PyTorch 1.9+  
- scikit-image  
- scikit-learn  
- NumPy  

### Install Dependencies
```bash
pip install torch torchvision scikit-image scikit-learn numpy
```

---

## 📁 Dataset Preparation

Organize your datasets as follows:

```text
root_dir/
├── dataset1/
│   ├── t1.png
│   ├── t2.png
│   └── ground_truth.tif
├── dataset2/
│   ├── ...
```

- Supported image types: `.tif`, `.png`, `.jpg`  
- Each dataset should contain:  
  - **T1 image** (pre-event)  
  - **T2 image** (post-event)  
  - **Ground truth image** (if available)  

---

## 🛠 Usage

### Training and Inference

Run the “France.py” script with appropriate arguments:

## 📈 Results Output

The method outputs:
- **Difference Image (DI)**  
- **Binary Change Image (BCI)** via Otsu thresholding  
- Evaluation metrics: **OA, Kappa, F1, AUR, AUP**  

IWCL outperforms multiple SOTA methods.  
For detailed results, please refer to **Tables 2–3** and **Figures 4–6** in our paper.

---

## 📜 Citation

If you use this code or our paper, please cite:

```bibtex
@ARTICLE{IWCL2026,
  author={Xu, Jianjian and Lei, Tao and Liu, Tongfei and Wang, Yingbo and Du, Xiaogang and Lv, Zhiyong and Gong, Maoguo},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={IWCL: Interactive Wavelet Commonality Learning for Unsupervised End-to-End Heterogeneous Change Detection}, 
  year={2026},
  volume={64},
  pages={1-14},
  doi={10.1109/TGRS.2026.3677107}
}

@article{ISPRS2025CGSL,
  title={CGSL: Commonality Graph Structure Learning for Unsupervised Multimodal Change Detection},
  author={Xu, Jianjian and Liu, Tongfei and Lei, Tao and Chen, Hongruixuan and Yokoya, Naoto and Lv, Zhiyong and Gong, Maoguo},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={229},
  pages={92--106},
  year={2025},
  publisher={Elsevier}
}

@article{TIP2025CFRL,
  author={Liu, Tongfei and Zhang, Mingyang and Gong, Maoguo and Zhang, Qingfu and Jiang, Fenlong and Zheng, Hanhong and Lu, Di},
  journal={IEEE Transactions on Image Processing}, 
  title={Commonality Feature Representation Learning for Unsupervised Multimodal Change Detection}, 
  year={2025},
  volume={34},
  number={},
  pages={1219-1233},
  keywords={Feature extraction;Image reconstruction;Training;Data mining;Autoencoders;Representation learning;Image sensors;Electronic mail;Decoding;Clustering algorithms;Multimodal change detection;unsupervised change detection;heterogeneous images;representation learning;commonality feature},
  doi={10.1109/TIP.2025.3539461}}
```

---

## ❤️Acknowledgement
We are very grateful for the outstanding contributions of the publicly available MCD datasets and codes [1,2,3] and [4].

```
[1] https://sites.google.com/view/luppino/data.
[2] Professor Michele Volpi's webpage at https://sites.google.com/site/michelevolpiresearch/home.
[3] Professor Max Mignotte's webpage (http://www-labs.iro.umontreal.ca/~mignotte/).
[4] https://github.com/yulisun.
```

---

## 📮 Contact

For questions or suggestions, please open an issue or contact:

- Jianjian Xu: xujianjian_leo@sust.edu.cn 
- Tongfei Liu: liutongfei_home@hotmail.com

**Note**: Please also copy to the two email addresses above.

---

