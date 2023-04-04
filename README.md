# PolarizedStacker
Polarized Snapshot Image Hierarchical Stacking

This package will read in a number of eht-imaging snapshot files and stack the posteriors together using an approximate hierarchical modeling scheme. Please see 
[EHTC IV SgrA*](https://iopscience.iop.org/article/10.3847/2041-8213/ac6736) for more information. 

## Installation

To install this package you need to first have a local version of Julia. To install Julia I recommend the juliaup package https://github.com/JuliaLang/juliaup and installing the Julia 1.8 series `juliaup add 1.8.5`. Once Julia is installed clone the repo and you should be good to go.

## Running the script

To run the stacker you need to run
```
juila -p NCORES main.jl directorylist priors_list.txt
```
where 
  
- `NCORES` is the number of cores you wish to use. Note we parallelize on the list of directories in `directorylist` so `NCORES` should be fewer than the number of directories
- `directorylist` is a file where each line is a path to the directory of the folder of the snapshot results. For instance if fitting two separate models `directorylist` would be
    ```
        /path/to/model1/directory
        /path/to/model2/directory
    ```
  an example can be found in `listdir`. **We recommend that all runs or data are place in the `Data/` folder.**
- `priors_list.txt` a text file with the list of priors and parameter names used. Two example priors are included (`priors_example_mring_m_stokesi_2_m_lp_3_add_floor=False.txt` and `priors_example_mring_m_stokesi_2_m_lp_3_m_cp_1_add_floor=False.txt`). Note that any "distance" variables e.g. diameter and width have to be in the same units as the chain files. For instance usually the chain files are in $\mu$as and so the priors also have to be in $\mu$as.

The output of the stacking will be in `directorylist/StackedResults` and will contain

1.  `stacker_chain.h5` which is a HDF5 file that groups all the snapshot results together
2.  `stacker_chain_ha_trunc.csv` which is a CSV file containing the chain of the results. Note this is the entire chain I would typically just recommend taking the last quarter of it for parameter inference.
3.  `stacker_chain_ckpt.jld2` is the checkpoint file for sampling that can be used to restart the script. 
