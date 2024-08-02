# AMT-GAF
Asymmetric Multi-Task system for Gaze-driven grasping Action Forecasting
### [Project Page](https://companion-cm.webs.tsc.uc3m.es/) | [Paper](https://ieeexplore.ieee.org/abstract/document/10602750)

> [Asymmetric Multi-Task system for Gaze-driven grasping Action Forecasting](https://ieeexplore.ieee.org/abstract/document/10602750)  
> Iván González Díaz, Miguel Molina-Moreno, Jenny Benois-Pineau and Aymar de Rugy
> IEEE Journal of Biomedical and Health Informatics, 2024 
> doi: 10.1109/JBHI.2024.3430810.

## Citation
If you find our code or paper useful, please consider citing our paper:
```BibTeX
@ARTICLE{Gonzalez-Diaz24,
  author={González-Diaz, Iván and Molina-Moreno, Miguel and Benois-Pineau, Jenny and de Rugy, Aymar},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Asymmetric multi-task learning for interpretable gaze-driven grasping action forecasting}, 
  year={2024},
  volume={},
  number={},
  pages={1-17},
  keywords={Task analysis;Predictive models;Visualization;Multitasking;Grasping;Forecasting;Hidden Markov models;Grasping action forecasting;multi-task learning;interpretable attention prediction;constrained loss},
  doi={10.1109/JBHI.2024.3430810}
 }
```

## HOW-TO-USE
We provide scripts for training and testing. You need to modify the paths in the config files located at experiments/config

### 1. SHARON dataset

The dataset can be downloaded from:
- SHARON-OBJECTS: to train object detectors with weak supervision (not necessary if you use our features). It can be downloaded from [here](https://www.tsc.uc3m.es/~igonzalez/resources/SHARON-OBJECTS.zip)
- SHARON-GRASP: for action forecasting. It can be downloaded from [here](https://www.tsc.uc3m.es/~igonzalez/resources/SHARON-GRASP.zip)

In addition, you can use our features for action forecasting (see our paper for the complete definition of features), which can be downloaded from [here](https://www.tsc.uc3m.es/~igonzalez/resources/SHARON-GRASP-features.zip) 

To train a new model, from the base folder AMT-GAF, type:

``` 
experiments/scripts/sharon_ap_cv.sh $fold_num $config_num
```

To test a model, from the base folder AMT-GAF, type:

``` 
experiments/scripts/sharon_ap_cvt.sh $fold_num $epoch_num $config_num
```

where $epoch_num is the epoch number you would like to test.

### 2. Grasping-in-the-wild dataset

To train a new model, from the base folder AMT-GAF, type:

``` 
experiments/scripts/gitw_ap_cv.sh $fold_num $config_num
```
To test a model, from the base folder AMT-GAF, type:

``` 
experiments/scripts/gitw_ap_cvt.sh $fold_num $epoch_num $config_num
```

where $epoch_num is the epoch number you would like to test.

### 2. Invisible dataset

To train a new model, from the base folder AMT-GAF, type:

``` 
experiments/scripts/invisible_ap_cv.sh $fold_num $config_num
```
To test a model, from the base folder AMT-GAF, type:

``` 
experiments/scripts/invisible_ap_cvt.sh $fold_num $epoch_num $config_num
```

where $epoch_num is the epoch number you would like to test.
