# Machine learning examples

Some of my notebooks made during ML discovering.

## 1. Anomaly and novelty detection (unsupervised)

```
#statistics #shapiro #threat-detection #decision-trees #lstm
```

### Objective

Find any unusualness like breach in the industrial system, anomaly activity like a fraud or other threats or novelty we want to detect. The data is a multivariable timeseries.

- Statistical approaches

![Shapiro test](https://lingtra.in/images/other/anomaly_detection_stat.png)

- Decision tree based methods

![DT ensembling](https://lingtra.in/images/other/anomaly_detection_dt.png)

- LSTM

![LSTM](https://lingtra.in/images/other/anomaly_detection_rnn.png)

Working notebook for experiments: https://www.kaggle.com/averkij/anomaly-detection-methods

## 2. Autoencoders

```
#conditional-autoencoders #variational-autoencoders #faces-generation
```

Autoencoders learn to discover the structure in the data and produce a compressed representation. Then it can try to restore the initial data from the compressed data. The main applications of autoencoders are:

- Image denoising
- Dimensionality reduction
- Image generation (for example adding a smile to the face, sunglasses, hat, etc.)

### Faces generation (Variational autoencoders)

![Faces generation](https://lingtra.in/images/other/faces_generation.png)

You can play with the faces generation example here:
https://www.kaggle.com/averkij/variational-autoencoder-and-faces-generation


### Numbers generation (Conditional vaiational autoencodes)

![cvae](https://lingtra.in/images/other/cvae-tsne.png)

You can play with the CVAE example here:
https://www.kaggle.com/averkij/conditional-variational-autoencoder-and-t-sne

## 3. Natural Language Processing

```
#nlp #pos-tagging #topic-modeling #machine-translation #word2vec #bpe
```

### Parts of speech tagging

![pos](https://lingtra.in/images/other/pos_tagging2.png)

- Example of how to do a POS tagging using different models:
  - Hidden Markov model
  - Stanford model

### Machine translation

![machine translation](https://lingtra.in/images/other/machine_translation.png)

Open-NMT based translation system for Chienese to Russian translation task.

You can see the details in this kernel: 
https://www.kaggle.com/averkij/opennmt-machine-translation-baseline

## 4. Recommender systems

```
#recsys #svd #als #item2item #user2item 
```

![recsys](https://www.researchgate.net/profile/Rajani_Chulyadyo/publication/312092950/figure/fig3/AS:457045389385734@1485979520674/3-Recommendation-approaches-a-Collaborative-filtering-b-Demographicsbased.png)

Simple example on recommender system using SVD dimensionality reduction.
