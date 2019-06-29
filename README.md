# alexnet-tensorflow-kaggle
AlexNet implementation with TensorFlow APIs (Kaggle)

# Usage

```alexnet.py``` is an CLI (Command Line Interface) application based on ```Python```.

Available options can be seen by running command ```python alexnet.py -h```.
```
~$ python alexnet.py -h
usage: alexnet.py [-h] [--train_p TRAIN_P] [--test_p TEST_P]
                  [--learning_rate LEARNING_RATE] [--epochs EPOCHS]
                  [--batch_size BATCH_SIZE] [--net_weights NET_WEIGHTS]
                  [--train] [--save_graph SAVE_GRAPH] [--labels LABELS]
                  [--infer INFER]

optional arguments:
  -h, --help            show this help message and exit
  --train_p TRAIN_P     path to train.p
  --test_p TEST_P       path to test.p
  --learning_rate LEARNING_RATE
                        learning rate for training
  --epochs EPOCHS       number of training iterations
  --batch_size BATCH_SIZE
                        batch size for training
  --net_weights NET_WEIGHTS
                        pretrained weights bvlc-alexnet.npy
  --train               train/retrain model
  --save_graph SAVE_GRAPH
                        saves graph as *.pb while training or loads *.pb for
                        inference
  --labels LABELS       labels file
  --infer INFER         perform inference for file
```

As seen in the provided help, this application can be used for training, retraining and inferencing. This also allows user to tune hyperparameters for training directly from Command Line.

Application takes train/test data in form of Pickle. In order to produce these, one can use ```dataset.py```.

```
~$ python dataset.py -h
usage: dataset.py [-h] [--csv_file CSV_FILE]

optional arguments:
  -h, --help           show this help message and exit
  --csv_file CSV_FILE  openimages dataset csv
```

## Dataset

To produce ```train.p``` and ```test.p``` from dataset ```*.csv```, one can run following command,

```
~$ python dataset.py --csv_file <path/to/dataset.csv>
```

For Kaggle requirement, ```openimage_5k.csv``` can be used as dataset from Open Image Dataset. This contains 5000 images.

## Training

In order to train alexnet for given data set, follow steps given below:

```
~$ python alexnet.py --train_p <path/to/train.p> --train --save_graph alexnet.pb --labels <path/to/labels.txt> --epochs 5 --batch_size 32 --learning_rate 0.001
```

Here, hyperparameters can be optional as internally they are assigned to some default values,

Default values for hyperparameters
```
epochs = 2
batch_size = 128
learning_rate = 0.001
```

## Inference

In order to use already trained model for inferencing,

```
~$ python alexnet.py --infer <path/to/image> --labels <path/to/labels.txt> --save_graph alexnet.pb
```