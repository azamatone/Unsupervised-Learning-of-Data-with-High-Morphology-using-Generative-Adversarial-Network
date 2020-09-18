# Unsupervised-Learning-of-Data-with-High-Morphology-using-Generative-Adversarial-Network
We propose Morpho-GAN, a method that unifies several GAN techniques to generate quality data of high morphology. Our method introduces a new suitable training objective in the discriminator of GAN to synthesize images that follow the distribution of the original dataset. The results demonstrate that the proposed method can generate plausible data as good as other modern baseline models while being a less complex during training.

Code accompanying our [Morpho-GAN](https://www.koreascience.or.kr/article/CFKO202015463051310.pdf), an award winning paper of 61st Winter Conference of Korean Society of Computer and Information.

## Setup

### Dependencies
Python 2.7 or 3.x, Numpy, scikit-learn, Tensorflow, Keras (1D demo) <br>
For Python 3.x - `import pickle` in the file `morphogan_image/modules/dataset.py`

### Getting Started
We conduct experiments of our model with proposed Morpho-GAN architecture. 

In case of other architecture, have your <br> 
`nnet_type` parameter set to the correct network type and 
`noise_dim` parameter set to the correct latent variable dimension for that network.

#### Image data (MNIST, CelebA and CIFAR-10)

We provide our code for image datasets Galaxy Zoo and optional MNIST.

##### Galaxy Zoo
Downloading Galaxy Zoo from: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data and extract into the correct folder: eg. `./data/galaxyzoo/`.

```
>> cd morphogan_image
>> python final_trainable_morpho.py.py
```

Have `db_name = 'galaxyzoo'` and `data_source = './data/galaxyzoo/'` parameters set as the correct dataset to train.

Generated samples are generated into a folder as `./morphogan_image/images/galaxyzoo_{}_fake.jpg`<br>
Real samples are generated into a folder as `./morphogan_image/images/galaxyzoo_{}_real.jpg`

##### MNIST


```
>> cd morphogan_image
>> python final_trainable_morpho.py.py
```

Generated samples are generated into a folder as `./morphogan_image/images/mnist_{}_fake.jpg`<br>
Real samples are generated into a folder as `./morphogan_image/images/mnist_{}_real.jpg`



## References

[1] Alec Radford, Luke Metz, Soumith Chintala, "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks" ICLR, 2017.<br>
[2] Sanjeev Arora, Rong Ge, Yingyu Liang, Tengyu Ma, Yi Zhang, "Generalization and equilibrium in generative adversarial nets (GANS)" ICML, 2017<br>
[3] Naveen Kodali, Jacob Abernethy, James Hays, Zsolt Kira "On Convergence and Stability of GANs" arXiv preprint arXiv:1705.07215, 2017.<br>
[4] Ngoc-Trung Tran, T. Bui, Ngai-Man Cheung, "Dist-GAN: An Improved GAN using Distance Constraints", ECCV, 2018.<br>
[5] J. D. Curtó, H. C. Zarza, T. Kim, "High-Resolution Deep Convolutional Generative Adversarial Networks", arXiv preprint arXiv: 1711.06491, 2017
