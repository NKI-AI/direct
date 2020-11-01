# DIRECT Model Zoo and Baselines

## Introduction
This file documents baselines created with the DIRECT project. You can find the parameters and weights of these models
in [projects/](projects/).

## How to read the tables
* "Name" refers to the name of the config file which is saved in `projects/{project_name}/configs/{name}.yaml`
* Checkpoint is the integer representing the model weights saved in `model_{iteration}.pt` as that iteration.
* Version refers to the corresponding [DIRECT release](https://github.com/directgroup/direct/releases).

## License
All models made available through this page are licensed under the
[Creative Commons Attribution-ShareAlike 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/).

## Baselines
### Calgary-Campinas baselines
There are two subchallenges: Track 01 and Track 02. In Track 01 the task is to reconstruct 12-coil data, and in Track 02
there is a generalization test to reconstruct 32-coil data. Both have two acceleration factors: 5x and 10x

#### Challenge set

##### Track 01 (12 channel)
| Name | Acceleration | Version | Checkpoint | SSIM  | pSNR | VIF   |
|------|--------------|---------|------------|-------|------|-------|
| base | 5x           | v0.2    | 88000      | 0.936 | 35.3 | 0.960 |
| base | 10x          | v0.2    | 80500      | 0.796 | 27.0 | 0.720 |


##### Track 02 (32 channel)
*Note: in Track 02 also 12 channel data was available, for which the same model as in Track 01 was used. Only metrics
for the 32 channel data are given here.*

| Name | Acceleration | Version | Checkpoint | SSIM  | pSNR | VIF   |
|------|--------------|---------|------------|-------|------|-------|
| base | 5x           | v0.2    | 88000      | 0.947 | 37.7 | 0.992 |
| base | 10x          | v0.2    | 80500      | 0.901 | 34.3 | 0.945 |

The models can be found in [projects/calgary_campinas/baseline_model]([projects/calgary_campinas/baseline_model]).
