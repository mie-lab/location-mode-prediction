# Next location prediction using transformers

This repository represents the implementation of the paper:

## [How do you go where? Improving next location prediction by learning travel mode information using transformers]()
[Ye Hong](https://scholar.google.com/citations?user=dnaRSnwAAAAJ&hl=en), [Henry Martin](https://n.ethz.ch/~martinhe/), [Martin Raubal](https://raubal.ethz.ch/)\
| [IKG, ETH Zurich](https://gis.ethz.ch/en/) | [IARAI](https://www.iarai.ac.at/) |

![flowchart](fig/main_flowchart.png?raw=True)

## Reproducing on Geolife dataset
The results in the paper are obtained from SBB Green Class and Yumuv dataset that are not publicly available. We provide a runnable example of the pipeline on the Geolife dataset. The travel mode of the Geolife users are determined through the provided mode labels and using the trackintel function `trackintel.analysis.labelling.predict_transport_mode`. The steps to run the pipeline are as follows:

### Install dependencies 
- Download the repo, install neccessary `Requirements and dependencies`.

### Download Geolife 
- Download the Geolife GPS tracking dataset from [here](https://www.microsoft.com/en-us/download/details.aspx?id=52367). Unzip and copy the `Data` folder into `geolife/`. The file structure should look like `geolife/Data/000/...`.
- Create file `paths.json`, and define your working directories by writing:

```json
{
    "raw_geolife": "./geolife",
}
```

### Preprocessing

### Run transformer

## Contact
If you have any questions, please let me know: 
- Ye Hong {hongy@ethz.ch}
