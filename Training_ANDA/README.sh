#ANDA

Please follow the https://github.com/NVlabs/stylegan2-ada-pytorch to build up the conda environment.



To train StyleGAN+ADA+ANDA on the 100-shot-obama dataset, run the following 

```
python train.py --outdir=training-runs --mirror=true --data=100-shot-obama.zip --gpus=1

```

It takes about several hours to finish the training. You can also change the --data in the above comments to obtain the results on other datasets. The results of FID values should be close to the reported FID in the paper.


