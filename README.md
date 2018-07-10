# ICDL-EpiRob 2018
This repository contains the related source --datasets, scripts, figures, etc.-- to reproduce the study submitted to ICDL-EpiRob2018.

> **Abstract:** This study reports our initial work on multimodal sensory representation for object classification. To form a sensory representation we used the spatial pooling phase of the Hierarchical Temporal Memory -- a Neocortically-inspired algorithm. The classification task carried out on Washington RGB-D dataset in which employed method provides extraction of non-hand engineered representations (or features)  from different modalities which are pixel values (RGB) and depth (D) information.  These representations, both early and lately fused,  used as inputs to a machine learning algorithm to perform classification. The obtained results show that using multimodal representations significantly improve (by $5 \%$) the classification performance while comparing single modality inputs. The results also indicate that the performed method is effective for multimodal learning and different sensory modalities are complementary for the object classification.  We, therefore, envision that this method can be employed for object concept formation that requires multiple sensory information to execute cognitive tasks.

## Folder descriptions
* **DATA:** The original dataset, its processed balanced version (494 RGB-D images per object) and the classification I/O data files. Moreover, in the same folder the confusion matrices, loss, and accurcy  values for training, validation and testing sets can be found.   
* **Figures:** The figures that used in the paper to show sample object and classification results.  
* **Source:** The collection of source code to process the dataset, extract representations and perform classification.  
* **Misc:** This folder contains various documents including the author copy of the previous study in the submitted paper.  
