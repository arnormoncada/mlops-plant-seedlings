# Project description

## Goal
This is the GitHub repository by Group 14. In this project, we develop an image classification model to classify images of crop and weed seedlings. The goal is to address a challenge in modern agriculture: improving efficiency and reducing labor while meeting increasing food demands.

Humans have been practicing agriculture for over 12,000 years and have continuously worked to increase both yield and efficiency. Over time, advancements in technology, methods, and tools have transformed the way we cultivate crops. However, one major challenge in agriculture today is meeting the growing global demand for food while grappling with limited resources, such as arable land, water, and labor. Weeds present a significant obstacle, as they compete with crops for essential nutrients and water, potentially reducing yields dramatically.

Weed removal is labor-intensive and often relies heavily on the use of herbicides, which can have long-term environmental and economic costs. The purpose of this project is to leverage machine learning to develop a model that can accurately distinguish between weed seedlings and crop seedlings. This model could serve as the foundation for autonomous systems capable of identifying and removing weeds efficiently. Such systems have the potential to revolutionize agriculture by reducing manual labor, minimizing herbicide use, and ultimately increasing crop yield in a sustainable manner.
## Framework
In this project, we utilize the TIMM (PyTorch Image Models) framework to access and implement image classification architectures. TIMM provides a wide range of pre-trained models, enabling us to select and fine-tune a suitable architecture for our specific classification task.
## Data
The data we plan to use is found at 
```shell
https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset
```
which has 12 classes with 5539 images. These classes consist of different common plant species in Denmark. In this dataset the classes have some imbalance. 
## Model 
For this project, we will use a pre-trained model from the TIMM package as our base architecture. Pre-trained models, such as ResNet, EfficientNet, or ConvNeXt
## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
