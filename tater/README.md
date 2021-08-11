# Running TATER

1. Open up the `run_tater.py` script and edit the TIC ID.

2. From the terminal, run `$ python run_tater.py`. Upon downloading the TESS data, the code may prompt you to enter information about the host star that was not automatically found on EXOFOP.

3. The code should proceed to download the TESS data, run TLS to detect TCEs, vet the TCEs to find planet candidates, and fit a transit model to each planet candidate.

### Notes:

- Make sure you are running Python 3.
- To avoid TLS errors, make sure you are using the version of TLS edited by CH.
- TATER should automatically generate and save files containing information from the fits.
