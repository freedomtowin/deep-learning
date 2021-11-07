# CNN Forecasting and Noise Clustering

This post is about the following:

1. TensorFlow 2 dataset pipelines with python function transformations

2. Background noise clustering (algorithm from IEEE Kaggle competition)

3. Simple CNN forecasting (with Fourier Transform features)


TF2 Datasets:

These create a HUGE improvement in code efficiency. However, a little bit of onboarding is required to create complex pipelines, i.e., using using custom python pre-processing functions or processing data-groups independently.



Background Nosie Source Code:

https://www.kaggle.com/zeemeen/i-have-a-clue-what-i-am-doing-noise-patterns

This is an interesting way to characterize the background noise in images (or time series?)

Forecasting:

I used my trusty stock dataset to test these concepts.

1. Find stocks with similar background noise
2. Build a data pipeline transform the time window with the Fourier transform
3. Forecast one of the stocks with the transformed features

[](.images/result.png)


