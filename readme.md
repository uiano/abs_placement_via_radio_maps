
Simulations for paper "Aerial Base Station Placement Leveraging Radio Tomographic Maps" by Daniel Romero, Viet Q. Pham, and Geert Leus.

https://arxiv.org/abs/2109.07372

## Instructions

Enter the folder where you have cloned this repository and do the following.

```
cd gsim
git submodule init
git submodule update
cd ..
```

Set whether you want to use Mayavi or not in gsim_conf.py. 

Browse the experiments in experiments/placement_using_channel_maps.py. Those in the paper are marked as "Conf. PAPER". 

To run experiment XXXX, type
```
 $ python run_experiment.py XXXX
```
where XXXX can be, for example, 1000 or 2023. 

You may need to install additional packages through pip. 
