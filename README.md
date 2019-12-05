## README

Material for Timeflux Workshops
------------------------------


## Installation
Download the zip or clone this repo: 
```
git clone https://github.com/bertrandlalo/timeflux_workshops
```

### If you already have a timeflux environment installed, then:

For the exercises: 
```
conda activate timeflux
pip install git+https://github.com/timeflux/timeflux_example
pip install git+https://github.com/timeflux/timeflux_ui
conda install jupyter
conda install ipython
conda install pygments
```
If you want to run the Use case on Oddball EEG ERP classification, you need to add:
```
conda install seaborn
pip install git+https://github.com/timeflux/timeflux_ml
pip install git+https://github.com/timeflux/timeflux_dsp
pip install pyriemann
```
To create the environment from scratch:
```
conda create env -f environment.yaml
conda activate timeflux-tutorials
```

## Authors

Raphaëlle Bertrand-Lalo, ... 

## License
[MIT](http://opensource.org/licenses/MIT).
