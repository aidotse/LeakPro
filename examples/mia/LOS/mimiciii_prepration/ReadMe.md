
## MIMIC-III Data Preparation
<br>This guide provides instructions for preparing the MIMIC-III dataset for a length of stay prediction example.

 ### Requirments
Ensure that your system has the following tools available in the PATH:

* conda
* [psql](https://www.postgresql.org/download/) (PostgreSQL 9.4 or higher)
* git
* Access the MIMIC-III Dataset

 ### Steps

 #### Create Conda envrioment
 Navigate to ```MIMIC_Extract``` and create a Conda environment:
``` 
conda env create --force -f mimic_extract_env_py36.yml
conda activate mimic_data_extraction
 ```

#### Compiling the data
Navigate to ```mimic-code/buildmimic``` . <br>
If you already have MIMIC-III dataset downloaded, place the unzipped CSV files in the  ```data ``` folder, and run the following check command:
``` bash
$ make mimic-check-csv datadir=./data/ 
```

If you have not yet downloaded the dataset, use the following command to download it (replace [your_username] with your PhysioNet username):

``` bash
$ make  mimic_download_check datadir=./data/ physionetuser=[your_username]
```

Then to create and build a postgres database run the below command. The default value for the postgres password is 'postgres'.
``` bash
$ make mimic-build datadir=./data/ 
```
Navigate back to the ```./MIMIC_Extract``` and run the follwoing command.
The output files will be saved in the ```output``` folder. 
```
$ bash run.sh
```

