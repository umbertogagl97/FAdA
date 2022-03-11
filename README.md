# FAdA: Fingerprint Adversarial Attacks in real world
<p align="center">
  <img src="logo224.png" />
</p>
FAdA is a Python library for Machine Learning Security based on Adversarial Robustness Toolbox (ART) and it is specialized in attacks to Fingerprint Liveness Detection.
The peculiarity of FAdA is that the attacks are aimed at the realization of presentation attacks that can be used in the real world

## Installation
- **Colab**: ```!pip install git+https://github.com/umbertogagl97/FAdA.git```
see examples/ScannerAttack for more informations

## fada
The main directory that contains:
- **utils.py**
libraries, transformations and functions for attacks implementation
- **attacks**
directory with main adversarial attacks
- **scanner**
notebooks for data augmentation and cnns training

## Examples
Directory that contains examples notebooks about attacks to scanners
- **ScannerAttack.ipynb** example of attacks aimed at the realization of presentation attacks that can be used in the real world

## How to start?
go to readme in "scanner" directory
