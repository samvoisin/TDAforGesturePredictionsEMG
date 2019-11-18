# Topological Data Analysis Techniques for Gesture Prediction

This research project seeks to use methods derived from topology to filter sEMG signals to improve predictive power for hand gestures.

Specifically, the aim of this project is to utilize topological and geometric methods for filtering surface Electromyography ("sEMG") signals derived from participants. These filtered signals will then be used in predicting the gestures a user is performing.


The data for this project was originally gathered through the study *Latent Factors Limiting the Performance of sEMG-Interfaces* by Lobov S., Krilova N. *et al.* published in *Sensors*. 2018;18(4):1122. doi: 10.3390/s18041122




Source data can be found here: https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures


To Do List:
1) Create data set of persistence diagram vectors X done
2) Add random Gaussian noise and recalculate persistence X done - null result
3) normalize persistence images for log reg X done
4) use persim to ID influential image components for filter
5) SSM fusion
6) CDER Classifier for time-series(?)
