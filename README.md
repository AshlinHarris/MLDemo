# MLDemo
Machine Learning with Synthetic Health Records

# Requirements
This tutorial was developed for the Synthea COVID-19 10K CSV data set: https://synthea.mitre.org/downloads.
The path of the directory containing these files needs to be saved in `config.ini`.

# Use
The script `main.jl` should run without issue after all dependencies are installed.
It produces the plots in the `Figures/` directory (needs to be specified locally in `config.ini`), but I have included these in the repository,
along with the standard output (`out.txt`).

# Future
Currently, the pipeline in `main.jl` pulls functions from `src/MLDemo.jl`.
All features are being integrated into the publicly-available Julia package: https://github.com/bcbi/PreprocessMD.jl
