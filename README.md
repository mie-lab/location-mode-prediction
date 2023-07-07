# Next location prediction using transformers

This repository represents the implementation of the paper:

## [How do you go where? Improving next location prediction by learning travel mode information using transformers](https://arxiv.org/abs/2210.04095)
[Ye Hong](https://hongyeehh.github.io/), [Henry Martin](https://n.ethz.ch/~martinhe/), [Martin Raubal](https://raubal.ethz.ch/)\
| [IKG, ETH Zurich](https://gis.ethz.ch/en/) | [IARAI](https://www.iarai.ac.at/) |

![flowchart](fig/main_flowchart.png?raw=True)

## Requirements and dependencies

This code has been tested on

- Python 3.9.12, Geopandas 0.12.2, trackintel 1.1.13, PyTorch 1.12.1, cudatoolkit 11.3, GeForce RTX 3090

To create a virtual environment and install the required dependences please run:
```shell
    git clone https://github.com/mie-lab/location-mode-prediction
    cd location-mode-prediction
    conda env create -f environment.yml
    conda activate loc-mode-pred
```
in your working folder.


## Hyperparameters
All hyperparameter settings are saved in the `.yml` files under the respective dataset folder under `config/`. For example, `config/geolife/transformer.yml` contains hyperparameter settings of the transformer model for the geolife dataset. 


## Reproducing on the Geolife dataset

The results in the paper are obtained from SBB Green Class and Yumuv dataset that are not publicly available. We provide a runnable example of the pipeline on the Geolife dataset. The travel mode of the Geolife users are determined through the provided mode labels and using the trackintel function `trackintel.analysis.labelling.predict_transport_mode`. The steps to run the pipeline are as follows:

### 1. Install dependencies 
- Download the repo, install neccessary `Requirements and dependencies`.

### 2. Download Geolife 
- Download the Geolife GPS tracking dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52367). Unzip and copy the `Data` folder into `geolife/`. The file structure should look like `geolife/Data/000/...`.
- Create file `paths.json`, and define your working directories by writing:

```json
{
    "raw_geolife": "./geolife"
}
```

### 3. Preprocess the dataset
- run 
```shell
    python prePro/geolife.py geolife 20
```
for executing the preprocessing script for the geolife dataset. The process takes 15-30min. `locations_geolife.csv` and `dataset_geolife.csv` will be created under `data/` folder.

### 4. Run the proposed transformer model
- run 
```shell
    python main.py config/geolife/transformer.yml
```
for starting the training process. The dataloader will create intermediate data files and saves them under `data/temp/` folder. The configuration of the current run, the network paramters and the performance indicators will be stored under `outputs/` folder.


## Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@inproceedings{Hong_2022_how,
    author      = {Hong, Ye and Martin, Henry and Raubal, Martin},
    title       = {How do you go where? Improving next location prediction by learning travel mode information using transformers},
    year        = {2022},
    doi         = {10.1145/3557915.3560996},
    booktitle   = {Proceedings of the 30th International Conference on Advances in Geographic Information Systems (SIGSPATIAL '22)},
    location    = {Seattle, Washington, USA},
    publisher   = {Association for Computing Machinery},
    address     = {New York, NY, USA},
}
```

## Contact
If you have any questions, please open an issue or let me know: 
- Ye Hong {hongy@ethz.ch}
