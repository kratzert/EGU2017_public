# Examples

This folder contains different scripts/notebooks to reproduce or better understand the results presented at the EGU 2017 General Assembly. Below are further information to each of the examples. Make sure to read them first!

## Example 1: explorer.ipynb

Within this notebook the test set results can be further analyzed. It's not a requirement to evaluate the different models on your own (see example 2 below), as the evaluation results for each of the 4 models can be found in the `eval/` folder. In case you have run the example 2 and want to expect the created result file, just make sure you paste the file in the `eval/` and change the filename in the notebook.
You'll find further instructions within the notebook.

### Requirements

If you want to run the notebook locally, you'll need the following packages (might run with previous versions, but tested with this specification):

- Python 3.5
- Numpy 1.12.1
- Pandas 0.1.2
- Jupyter 1.0.0
- Scikit-learn 0.18.1
- Scikit-image 0.13.0
- Matplotlib 2.0.0


## Example 2: eval.py

This script can be used to evaluate the different models against the test set (or sample test set). The results will be saved into a .csv file in the `examples` folder. From the terminal you start the evaluation by

```terminal
$ python eval.py model -m=mode
```
where `model` is one of `{vgg, vgg_w_length, vgg_w_date, vgg_w_all}` and `mode` one of `{sample/all}`. The `mode` argument is optional, where `sample` is the default.

E.g. if you want to evaluate the model that combines the images with the length and date feature on the entire test set you can start the script by

```
$ python eval.py vgg_w_all -m=all
```

Another example would be to evaluate the model that combines images with length on the sample test set. This could be started by

```
$ python eval.py vgg_w_length
```

**Note**: To evaluate any of the models, you have to download the model specific weight files of the network (~670 MB). The different files can be found here:

- vgg (image only): [Link](https://drive.google.com/drive/folders/0B3YsW-PFiJOLWEVzbUtkcEo5U2s?usp=sharing)
- vgg_w_length (image + length): [Link](https://drive.google.com/drive/folders/0B3YsW-PFiJOLQUVUYWd1cllFVE0?usp=sharing)
- vgg_w_date (image + date): [Link](https://drive.google.com/drive/folders/0B3YsW-PFiJOLRURzOURKc1NlOGs?usp=sharing)
- vgg_w_all (image + both): [Link](https://drive.google.com/drive/folders/0B3YsW-PFiJOLcHBOTkJFNmd6NDg?usp=sharing)

For any model you want to test you have to download the two files which you can find in each of the links. Within the `examples` folder you have to create a new folder called `checkpoints` in which you have to create one folder for each of the models you want to test with the same name as the model.
- E.g. `../examples/checkpoints/vgg_w_all/` and copy both of the `vgg_w_all` files at this location.

**Note2**: This is a computational expensive task. Without GPU acceleration the sample test set evaluation might take some minutes to finish.

### Requirements

You can use this script with either Python2 or Python3. It's recommended to use a virtualenvironment (conda environment), when testing this script.

The Python3 requirements are:
- Python 3.5
- Numpy 1.12.1
- Pandas 0.19.2
- Scikit-learn 0.18.1
- Scipy 0.19.1
- Tensorflow 1.0

The Python2 requirements are:
- Python 2.7.13
- Numpy 1.12.1
- Pandas 0.19.2
- Scikit-learn 0.18.1
- Scipy 0.19.0
- Tensorflow 1.0
