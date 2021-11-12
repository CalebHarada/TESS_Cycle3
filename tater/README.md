# Running TATER

0. Download the `main.py` script located in the `tls_files` folder and use this to replace the existing script in your local `transitleastsquares` install directory.

1. Open up the `run_tater.py` script and edit the TIC ID.

2. From the terminal, run `$ python run_tater.py`. Upon downloading the TESS data, the code may prompt you to enter information about the host star that was not automatically found on MAST.

3. The code should proceed to download the TESS data, run TLS to detect TCEs, vet the TCEs to find planet candidates, and fit a transit model to each planet candidate.

### Notes:

- The edited version of `main.py` is needed to resolve a bug in TLS --- this has to do with changes in cadence and large gaps in observations for TESS data.
- Make sure you are running Python 3.7. For some reason, multiprocessing in Python 3.8 is buggy on MacOS
- TATER should automatically generate and save files containing information from the fits.
