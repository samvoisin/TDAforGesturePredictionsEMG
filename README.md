# Topological Data Analysis Techniques for Gesture Prediction

This research project seeks to adapt data analysis methods derived from topology and high-dimensional geometry to electromyogram data for classification purposes.

Specifically, the aim of this project is to precisely describe the topological and geometric properties of a high-dimensional manifold sampled by an array of surface Electromyography ("sEMG") sensors as a subject performs a series of predetermined movements.

The data for this project was originally gathered through the study *Latent Factors Limiting the Performance of sEMG-Interfaces* by Lobov S., Krilova N. *et al.* published in *Sensors*. 2018;18(4):1122. doi: 10.3390/s18041122




Source data can be found here: https://archive.ics.uci.edu/ml/datasets/EMG+data+for+gestures


To Do List:
1) Create data set of persistence diagram vectors X done
2) Add random Gaussian noise and recalculate persistence X done - null result
3) normalize persistence images for log reg X done
4) use persim to ID influential image components for filter X done
5) SNF X done

6) Complete persistence filter development - in progress
7) Compare fused similarity templates across gesture classes and within and across subjects
8) Compare fused similarity template variance (static vs clan images) to variance in modalities
    - How does variance within and between modalities effect SNF template outcome
9) Scattering transform 
