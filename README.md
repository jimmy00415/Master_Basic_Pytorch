# Master Basic PyTorch — COMP3057 Lab Collection

A curated set of Jupyter notebooks to learn PyTorch from tensors and training loops to CNNs and LSTMs. Ordered for review with numeric prefixes (01_..19_).

## Repository structure

- 01_Introduction_to_Data-students.ipynb
- 02_Class_Exercise_Images_as_Data.ipynb
- 03_pytorch_basics_student.ipynb
- 04_Train_an_AI_model.ipynb
- 05_Exercise_LinearLayer_Learn_.ipynb
- 06_Relu.ipynb
- 07_One_Hot_Encoding_PyTorch.ipynb
- 08_Basic_Operation_DL_Convolution_MultiFilter_EN - Student.ipynb
- 08_Basic_Operation_DL_Convolution_MultiFilter_EN - Student.md
- 09_CNN_Architecture_Students.ipynb
- 10_lenet_tutorial.ipynb
- 11_VGGALEXNETRESNET_student.ipynb
- 12_Lecture_Demo_Week_7_Students.ipynb
- 13_rnn_training_student.ipynb
- 14_madrid_air_quality_lstm.ipynb
- 15_madrid_air_quality_lstm .ipynb (note: filename has a trailing space before .ipynb)
- 16_COMP3057_Assignment_Data-students.ipynb
- 17_Assignment2.ipynb
- 18_Homework1.ipynb
- 19_Lab_Demo_2.ipynb
- STUDY_GUIDE_COMP3057.md — course-wide study guide + practice checklist

Data:
- data/MNIST/raw/*
- data/FashionMNIST/raw/*

## Setup

Requirements: Python 3.9+ (or 3.10/3.11), PyTorch, torchvision, matplotlib, numpy.

### Quick start (conda)

```powershell
# Create env (optional)
conda create -n pytorch-basic python=3.10 -y; conda activate pytorch-basic

# Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or with CUDA (adjust version for your GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Common libs
pip install matplotlib numpy
```

Open the notebooks in VS Code or Jupyter and run cells top-to-bottom. The files are numbered to guide your learning path.

## Learning path (high level)

1) Data basics: tensors, splits, normalization
2) Images as data: NCHW, transforms, dataloaders
3) PyTorch basics: autograd, modules, training loop
4) Training practice: losses, metrics, LR
5) Linear regression demo (y=2x+1): MSE + SGD
6) ReLU nonlinearity and variants
7) One-hot vs class indices (CrossEntropy)
8) Convolutions and multi-filters; pooling
9) CNN blocks and regularization
10) LeNet on MNIST-like data
11) VGG/AlexNet/ResNet concepts
12) Training dynamics and debugging
13) RNN training patterns
14–15) LSTM for time series (Madrid air quality)
16–19) Assignments, homework, and labs

## Notes

- 08 has an accompanying explainer: `08_Basic_Operation_DL_Convolution_MultiFilter_EN - Student.md`.
- One notebook has a trailing space in the filename (`15_madrid_air_quality_lstm .ipynb`). You can rename to `15_madrid_air_quality_lstm.ipynb` later for consistency.

## License

Educational use. Add a LICENSE file if you need a specific license.
