
## MIMIC-III Data Preparation
<br>This is an instriction for prepratin MIMIC-III for the lenght of stay preditcion example. 

 ### Requirments
Your system should have the following executables on the PATH:

* conda
* [psql](https://www.postgresql.org/download/) (PostgreSQL 9.4 or higher)
* git
* Access the MIMIC-III Dataset

 ### Steps

 #### Create Conda envrioment 
``` 
conda env create --force -f ../mimic_extract_env_py36.yml
conda activate mimic_data_extraction
 ```

#### Compiling the data
If you have MIMIC-III dataset downloaded put the zip files in ```./mimic-code/buildmimic/data ```. Otherwise the following scripts will download the files. Then it unzip the file, create a postgres database, build the database and then create the needed concspets. For that redirect to  ```./mimic-code/buildmimic ``` and run the following command:
``` bash
$ make create-user mimic-build datadir=./data/ 
```
Then redirect to ./MIMIC_Extract and run the follwoing command, and then you should find the output files in ```output ``` folder. 

```
$ bash run.sh
```
<!-- /home/fazeleh/cleaning_data/mimic-code/mimic-iii/buildmimic/postgres
``` -->
