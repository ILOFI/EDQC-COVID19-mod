# **EDQC-COVID19-mod**

A  modified SEIR model which introduced dynamic parameters to incoporate the effects of non-pharmaceutical intervention (NPIs) for COVID-19.

## Requirements

The following Python packages are needed for running scripts (tested versions of packages are listed in parenthesis, based on Python 3.6.0):

- numpy (1.11.3)
- pandas (0.19.2)
- matplotlib (2.0.0)
- pymc (2.3.6)

## Simulating scenarios with or without some of NPIs

Run `model_simulation.py` after all requirements installed. Both figures and CSV data would be generated in `./simuresult/` for the main analysis and in `./sensitivity_result/` for sensitivity analyses within minutes.

- Figures are produced in PDF format in `./simuresult/` with the corresponding names.
- The detailed results are listed as CSV files, which are named by the format `resultsimuXYZ.csv`. `X, Y, Z` are dummy variables indicating whether a targeted intervention is implemented (1) or not (0).
  - `X` represents the implementation of localized lockdown.
  - `Y` represents the implementation of close-contact tracing.
  - `Z` represents the implementation of community-based nucleic acid testing (NAT) .
  - For example, the result file for scenario without any NPIs is `resultsimu000.csv`; the result file for scenario with only intervention of close-contact tracing is `resultsimu010.csv`.
- For each CSV file, the following columns are contained
  - `Date`  specifies the date of rows.
  - `reported_I` actually reported number of cumulative cases by date of onset.
  - `incidence_conbined_intervention_I`  estimated number of new cases with all NPIs implemented, *i.e.*, the actual scenario in Beijing.
  - `incidence_simulation_I` estimated number of new cases under simulation scenario with or without some of NPIs, which is defined by the name of CSV.
  - Both `incidence_conbined_intervention_I` and `incidence_simulation_I` are median value of the simulation, the 95% CI is also provided in `incidence_conbined_intervention_I_up(/down)` and  `incidence_simulation_I(/down)`.

Run `sampling_main.py` for parameter sampling in main analysis by MCMC approach, and `sampling_Sn.py` for sampling in sensitivity analyses ($n\in{1,2,3,4,5,6}$). The sampling of parameter will be saved to the **Folder ``objs``**. 
*Since it takes a long time to re-estimate the parameters, this repository contains the parameters corresponding to the results in the paper, and the step is optional.*

## Structure of this repository 

- `./modeldata/` contains necessary data for model.
- `./objs/` contains parameters of the model.
- `./simuresult/` contains output of simulation.
- `./sensitivity_result/` contains output of sensitivity analyses.
- `./model_simulation.py` the main code of simulation.
- `./sampling_{scenario}.py` the code for parameter sampling in different settings of model.

## License

This repository is licensed under the GNU General Public License (GPL).