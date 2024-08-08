# Evaluation Codes and Pre-trained models of WACV2024 paper ''Improving the Leaking of Augmentations in Data-Efficient GANs via Adaptive Negative Data Augmentation''

We have provided the pre-trained models of ANDA with Different DE-GANs on low-shot datasets for better obtaining the results we reported in the paper. The code of this module is built by ourselves based on the test codes of the DiffAug-GAN [[link]](https://github.com/mit-han-lab/data-efficient-gans) and ADA [[link]](https://github.com/NVlabs/stylegan2-ada-pytorch). 

# Dataset

The low-shot datasets can be found in [[link]](https://drive.google.com/file/d/1rWqaVlms55604jrP5t9ShacL6mZKWL8f/view?usp=sharing)

# Requirement

Please follow the DiffAug-GAN [[link]](https://github.com/mit-han-lab/data-efficient-gans) and ADA [[link]](https://github.com/NVlabs/stylegan2-ada-pytorch) to build the envirment required. 

# Pre-trained models of ANDA with StyleGAN2 + DiffAugment on Low-shot datasets

Pre-trained model on 100-shot-obama dataset [[link]](https://drive.google.com/file/d/1gGNKasAsnDbBJ01h40s8x4KN-jmrKda7/view?usp=sharing)

Pre-trained model on 100-shot-panda dataset [[link]](https://drive.google.com/file/d/1t7LkDajXx_Mf49Sp4dRDlaZVCxXd7CSs/view?usp=sharing)

Pre-trained model on 100-shot-grumpy_cat dataset [[link]](https://drive.google.com/file/d/1wrWRgh-l-KsRtX8P22QVK3K2Ub8SY3nC/view?usp=sharing)

Pre-trained model on AnimalFace-cat dataset [[link]](https://drive.google.com/file/d/1mb6wZaEg-rybVVG3PDyYe8sFAq17k6dE/view?usp=sharing)

Pre-trained model on AnimalFace-dog dataset [[link]](https://drive.google.com/file/d/1RaDkC2Y0jwIAHbwSBDwgSG1N9VEvJoat/view?usp=sharing)

# Pre-trained models of ANDA with StyleGAN2 + ADA on Low-shot datasets

Pre-trained model on 100-shot-obama dataset [[link]](https://drive.google.com/file/d/1WBVWypVyUp4Qg9WAhquo7Qgp3WAbwYuI/view?usp=sharing)

Pre-trained model on 100-shot-panda dataset [[link]](https://drive.google.com/file/d/1MaQjmb_mlsQfbuQtQxrLkVXmHgXwj-A_/view?usp=sharing)

Pre-trained model on 100-shot-grumpy_cat dataset [[link]](https://drive.google.com/file/d/1Ste68t4umvRtcR2lSrv_yqDrkkp85yus/view?usp=sharing)

Pre-trained model on AnimalFace-cat dataset [[link]](https://drive.google.com/file/d/1zv6zmlcuc4G8SjT-iyn28AREy327WmxK/view?usp=sharing)

Pre-trained model on AnimalFace-dog dataset [[link]](https://drive.google.com/file/d/1x5dS4mLy4dIga8GZNvYY938ClGR6rEQY/view?usp=sharing)

# Pre-trained models of ANDA with InsGen on Low-shot datasets

Pre-trained model on 100-shot-obama dataset [[link]](https://drive.google.com/file/d/1MAKPfPNzPdrDhkCwqZEBJuZDAZq7ebIU/view?usp=sharing)

Pre-trained model on 100-shot-panda dataset [[link]](https://drive.google.com/file/d/1aM3G17Aqzvh2D-z45RvDt0znczjI_wXn/view?usp=sharing)

Pre-trained model on 100-shot-grumpy_cat dataset [[link]](https://drive.google.com/file/d/1hu-SUNIlKdrNSeJMueYyeeG75ysHJGUE/view?usp=sharing)

Pre-trained model on AnimalFace-cat dataset [[link]](https://drive.google.com/file/d/1oHWmnwqB-ZF_Y0RfUJ3nyMe0DxfnZBbc/view?usp=sharing)

Pre-trained model on AnimalFace-dog dataset [[link]](https://drive.google.com/file/d/1h4hFqavloaev34y2dwTgKa1n4A5ErdZm/view?usp=sharing)

# Evaluation

To evaluate the Pre-trained models on low-shot datasets, run the following command:

```
python calc_metrics.py --metrics=fid50k_full --data=<which-dataset> --network=<which-pretrained>
```

To generate images using the Pre-trained models, run the following command:

```
python generate.py --outdir=out --seeds=1-16 --network=<which-pretrained>
```

To generate gifs using the Pre-trained models, run the following command:

```
python generate_gif.py --output=<which-dataset>.gif --seed=0 --num-rows=1 --num-cols=8 --network=<which-pretrained>
```

Please note that we evaluate all the pre-trained models on a Alienware R8 desktop with ubuntu 20.04 with an NVIDIA 2080TI GPU. The FID will be slightly different (slightly better or worse) if you apply different NVIDIA GPU and different system to evaluate the pre-trained models.

The training codes will be released after the journal submission is finished.

# Citation:
```
@inproceedings{zhang2024improving,
  title={Improving the Leaking of Augmentations in Data-Efficient GANs via Adaptive Negative Data Augmentation},
  author={Zhang, Zhaoyu and Hua, Yang and Sun, Guanxiong and Wang, Hui and McLoone, Se{\'a}n},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5412--5421},
  year={2024}
}
