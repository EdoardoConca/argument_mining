# Argument Mining in Healthcare Domain
This repository aims to reproduce the work of [Enhancing evidence-based medicine with natural language argumentative analysis of clinical trials](https://www.sciencedirect.com/science/article/abs/pii/S0933365721000919) by [Tobias Mayer et al.] , incorporating key improvements suggested by the authors as part of a deep learning project.

More specifically, the Argument Component model is made up of two components:
- The Argument Component Detector (ACD) model: it aims to detect the argument components in the text and classify them into Claims and Premises.
- The Relation Classifier (RC) model: it aims to classify the relation between the argument components into one of the following categories: Relation or NoRelation.

The dataset used for this project is the [AbstRCT Argument Mining Dataset](https://gitlab.com/tomaye/abstrct) which is a collection of argumentative texts in the healthcare domain.

## Folder Structure
```
|-- models/
|   |-- ac_detector.py
|   |-- relation_classifier.py
|-- data/
|   |-- dev/
|   |-- test/
|   |-- train/ 
|-- notebooks/
|   |-- arguments_component_detection.ipynb  
|   |-- relation_classification.ipynb          
|-- README.md
```