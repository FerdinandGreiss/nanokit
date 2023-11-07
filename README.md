# nanokit

Process movies and analyse protein synthesis spots from single DNA molecules immobilized on a surface. 

This repository contains two Python scripts to perform these jobs:

1) **drift-correction**: Correct lateral drift in fluorescence movies using fiducial markers (e.g. TetraSpeck microspheres)
2) **data-analysis**: Analyze protein expression spots from surface-immobilized DNA molecules with *E. coli* cell lysate

Detailed descriptions of how to run and what output these scripts generate are given in the Python files.

### Experimental raw data

The corresponding raw data to use this code for is provided at [Zenodo](https://zenodo.org/doi/10.5281/zenodo.8359539).

### Installation and systems requirements

I recommend to install the required packages with [Conda](https://docs.conda.io/en/latest/). 

This code for drift correction was tested on a personal Apple laptop and Windows computer. The data analysis was only tested on a personal Apple laptop (Mac OS 11.2.2), programmed with Visual Studio Code v1.82.2, and run with Python 3.7.3.

### Instructions

Detailed instructions to run the code are provided in the Python files as multi-line docstring. More specific explanations of the code is given as comments along the code. On a personal laptop, the drift correction usually runs for ~1 min; the data analysis can run for ~5 mins. 

### Contributions

If you have suggestions, questions, or any other reason to contact me, ferdinand.greiss@weizmann.ac.il

### Credits

Some parts of the scripts were adapted from the two amazing projects
- https://github.com/alecheckert/quot
- https://github.com/jungmannlab/picasso
