# Traffic Sign Recognition

![Traffic Sign Image](https://static.designandreuse.com/news_img17/20170103b_1.jpg)

## Overview

The Traffic Sign Recognition project uses a Convolutional Neural Network (CNN) to recognize traffic signs from images. Traffic sign recognition is crucial for numerous applications including autonomous driving systems and advanced driver assistance systems (ADAS). This project is based on the Traffic Sign Benchmark held at the IJCNN 2011.

## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Processing](#data-processing)
- [Model Building](#model-building)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The Traffic Sign Benchmark consists of:

- Single-image, multi-class classification problem.
- More than 40 classes.
- Over 50,000 images in total.
- A large, lifelike database.

The dataset for this project can be accessed on Kaggle (please insert the actual Kaggle link here as it wasn't provided).

Ensure you have downloaded the dataset and placed it in the project directory before proceeding.

## Installation

1. Clone the repository:

```bash
git clone <https://github.com/qusaikarrar/Traffic-Sign-Recognition.git>
cd <Traffic-Sign-Recognition>
```

## Usage

Execute the provided Python script or Jupyter Notebook to train the model, visualize results, and evaluate performance.

## Data Processing

The data is loaded from the provided CSV files and is processed as follows:
- Images are resized to 30x30 pixels.
- Images are converted to NumPy arrays for processing.
- The dataset is split into training and testing subsets.
- Labels are converted to one-hot encoded format.

## Model Building

The architecture of the CNN model consists of:
- Two initial convolutional layers with 32 filters and a kernel size of 5x5.
- Max pooling to reduce spatial dimensions.
- Two convolutional layers with 64 filters and a kernel size of 3x3.
- A fully connected layer with 256 nodes.
- The final layer with 43 nodes (representing each traffic sign class) using softmax activation for classification.

## Training

The model is subjected to the following training parameters and practices:
- Trained using the Adam optimizer.
- Categorical cross-entropy is used as the loss function.
- A total of 15 epochs, with the training history stored for subsequent analysis.

## Evaluation

Key points on the model evaluation:
- The performance of the model is visualized using accuracy and loss plots.
- The final model is evaluated on the test dataset, and its accuracy is printed.

## Model Summary

A brief insight into the model:
- Utilizes convolutional layers to efficiently extract image features.
- Pooling layers are employed to reduce spatial dimensions of the features.
- Dense layers handle the classification tasks.
- Dropout layers are strategically placed to prevent overfitting by randomly setting a fraction of input units to 0 during training updates.

For an in-depth model summary, inclusive of layer shapes and parameters, kindly refer to the provided code.

## Contributing

Contributions to this project are heartily welcome. For any suggestions, improvements, or bugs identified, feel free to open an issue or initiate a pull request.

## License

This project is graciously licensed for free use.
