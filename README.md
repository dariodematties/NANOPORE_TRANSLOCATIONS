# NANOPORE TRANSLOCATIONS SIGNAL PROJECT

In this project there are Machine Learning implementation workflows devoted to classification, regresion and detection of translocations events and statistics
produced in nanopore translocation traces.

For instance, in this repository we have three main components.
* A Translocation Counter
* A Feature Predictor
* A Translocation Detector

To obtain access to the datasets to train, validate and test the models, please go to Zenodo:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5013856.svg)](https://doi.org/10.5281/zenodo.5013856)

There you will be able to find
* All the already trained and best validated models for all SNRs
* The test datasets for all the SNRs and
* The train, validation and test datasets only for SNR=4

The code of this project is also cited on Zenodo:
[![DOI](https://zenodo.org/badge/321695864.svg)](https://zenodo.org/badge/latestdoi/321695864)

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

## Nanopore Translocation Detector

This is a Pulse dEtection TRansformer (PETR), for pulse detection. PETR determines the start and end time points of individual pulses, thereby singling out pulse segments in a time-sequential trace.
The Transformer in this network uses the pre-trained *Feature Predictor* as Backbone.
Basically the Backbone was fine-tuned for the detection task of the Transformer.

To evaluate the *Translocation Detector* you must go to `Detector_Validation` folder.

Want to see the help?

run

`main.py -h`

This is only for plotting the training history of the model (Loss and validation errors) --plot-training_history

`main.py --cpu --num-queries 75 -pth -v --resume DETR_ResNet18/checkpoint.pth.tar ../../Datasets/SNR_4/ ../Translocations_Counter/ResNet18/SNR_4/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_4/model_best.pth.tar`

This is only for plotting some detection examples of the trained model

`main.py --cpu --num-queries 75 -pth -v --run --run-plot-window 0.5 --batch-size 4 ../../Datasets/SNR_4/ ../Translocations_Counter/ResNet18/SNR_4/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_4/model_best.pth.tar ../Translocations_Detector/DETR_ResNet18/Confirmation_Model/model_best.pth.tar`

This is only for computing error stats of the trained model

`main.py --cpu --num-queries 75 -v --statistics -save-stats ./STATS_SNR_4/ --start-threshold 100 --end-threshold 400 --step-threshold 10 ../../Datasets/SNR_4/ ../Translocations_Counter/ResNet18/SNR_4/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_4/model_best.pth.tar ../Translocations_Detector/DETR_ResNet18/SNR_4/model_best.pth.tar`

This is only for computing and saving the outputs from a trained model

`main.py --cpu --num-queries 75 -v -save-outputs ./OUTPUTS_SNR_4/ ../../Datasets/SNR_4/ ../Translocations_Counter/ResNet18/SNR_4/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_4/model_best.pth.tar ../Translocations_Detector/DETR_ResNet18/SNR_4/model_best.pth.tar`

This is only for plotting error stats of the trained model from a file

`main.py --cpu --num-queries 75 -v -stats-from-file ./STATS_SNR_4/DETR_ResNet18/stats.pth.tar ../../Datasets/SNR_4/ ../Translocations_Counter/ResNet18/SNR_4/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_4/model_best.pth.tar ../Translocations_Detector/DETR_ResNet18/SNR_4/model_best.pth.tar`

This is only for plotting some detection examples of the trained model on experimental data set

```
main_rd.py --cpu --num-queries 75 -v --run --run-plot-window 1.0 --trace_number 5 --window_number 6 ../../Datasets/Real_Data/Lambda_DNA_Pos/ ../Translocations_Counter/ResNet18/SNR_2/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_2/model_best.pth.tar ../Translocations_Detector/DETR_ResNet18/SNR_2/model_best.pth.tar

main_rd.py --cpu --num-queries 75 -v --run --run-plot-window 0.2 --trace_number 0 --window_number 2 ../../Datasets/Real_Data/Streptavidin_Pos/ ../Translocations_Counter/ResNet18/SNR_1/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_1/model_best.pth.tar ../Translocations_Detector/DETR_ResNet18/SNR_1/model_best.pth.tar
```

This is only for computing and saving the predictions generated by the detector on the real data set

```
main_rd.py --cpu --num-queries 75 -v --compute-predictions ./SNR_2_MODEL_PREDICTIONS_Lambda_DNA_Pos/ ../../Datasets/Real_Data/Lambda_DNA_Pos/ ../Translocations_Counter/ResNet18/SNR_2/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_2/model_best.pth.tar ../Translocations_Detector/DETR_ResNet18/SNR_2/model_best.pth.tar

main_rd.py --cpu --num-queries 75 -v --compute-predictions ./SNR_1_MODEL_PREDICTIONS_Streptavidin_Pos/ ../../Datasets/Real_Data/Streptavidin_Pos/ ../Translocations_Counter/ResNet18/SNR_1/model_best.pth.tar ../Feature_Prediction/ResNet18/SNR_1/model_best.pth.tar ../Translocations_Detector/DETR_ResNet18/SNR_1/model_best.pth.tar
```
