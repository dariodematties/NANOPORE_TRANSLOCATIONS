# NANOPORE TRANSLOCATIONS SIGNAL PROJECT

In this project there are Machine Learning implementation workflows devoted to classification, regresion and detection of translocations events and statistics
produced in nanopore translocation traces.

For instance, in this repository we have three main components.
* A Translocation Counter
* A Feature Predictor
* A Translocation Detector

## Nanopore Translocations Counter

This stage of the project implements a network that counts the number of translocation events in a temporal window.
A temporal window is just a temporal chunk extracted from a trace.
To use this, go to the `Translocations_Counter` directory.

Run `main.py -h` to see the help.

Want to train a model from scratch?

run

`main.py --print-freq 500 --optimizer sgd --lr 0.001 --lrsp 10 -b 32 -v ../../Datasets/SNR_4/`

Want to train a model from a checkpoint?

run

`main.py --print-freq 500 --optimizer sgd --lr 0.001 --lrsp 10 -b 32 -v --resume ./ResNet18/checkpoint.pth.tar ../../Datasets/SNR_4/`

Want to only evaluate a trained model?

run

`main.py --resume ./ResNet18/model_best.pth.tar --print-freq 500 -e -v ../../Datasets/SNR_4/`

To plot the training history od the model run

`main.py --resume ./ResNet18/SNR_4/checkpoint.pth.tar -pth -v ../../Datasets/SNR_4/` or run just

`run main.py --resume ./ResNet18/SNR_4/model_best.pth.tar -pth -v ../../Datasets/SNR_4/`

if you want to show the training history until the moment when the best model was saved.

## Nanopore Translocations Feature Predictor

This stage of the project implements a network that predicts the average *duration* and *amplitude* of all translocations event in a temporal window.

Please go to `Feature_Prediction` folder to use this program.

Run `main.py -h` to see the help.

The command line layout is very similar to the previous one.

Want to train a model from scratch?

run

`main.py --print-freq 500 --optimizer sgd --lr 0.001 --lrsp 20 -b 32 -v ../../Datasets/SNR_4/`

Want to train a model from a checkpoint?

run

`main.py --print-freq 500 --optimizer sgd --lr 0.001 --lrsp 20 -b 32 -v --resume ./ResNet18/checkpoint.pth.tar ../../Datasets/SNR_4/`

and so forth.

You can evaluate the previous two stages acting in tandem.

## Backbone Validation

To evaluate the *Translocation Counter* and the *Feature Predictor* acting in tandem you must go to `Backbone_Validation` folder.

Want to see the help?

run

`main.py -h`

The following command is for computing and saving stats (errors-standard deviations) of a trained model and plot them for each situation.
A situation is a combination of several parameters in the dataset which characterize a temporal window.
For instance, a combination of a value of `Cnp` concentration of nanoparticles, `Dnp` diameter of nanoparticles and translocation duration is a situation.

`main.py -stats -save-stats ./STATS_SNR_4/ -v ../../Datasets/SNR_4 ../Translocations_Counter/ResNet18/SNR_4/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_4/model_best.pth.tar`


This is only for plotting stats (errors-standard deviations) which are loaded from a file whose path is provided as an argument.

`main.py --cpu -stats-from-file ./STATS_SNR_4/ResNet18/stats.pth.tar -v ../../Datasets/SNR_4 ../Translocations_Counter/ResNet18/SNR_4/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_4/model_best.pth.tar`


Finally this command is only for running a trained backbone and showing some plots of its prediction performance on noisy signals.

`main.py --run -b 5 -v ../../Datasets/SNR_4/ ../Translocations_Counter/ResNet18/SNR_4/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_4/model_best.pth.tar`
