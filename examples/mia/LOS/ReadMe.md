#  MIMIC-III Data Preprocessing
This document is a based on this and that.
The aim is to load MIMIC-III dataset and preprosesse it according to for following tasks:

**Binary Classifications:**

1- LOS > 3 Days

2- LOS > 7 Days

3- In-Hospital Mortality

4- In-ICU Mortality

**Multi class classficiation:**

1- Clinical(mechanical ventilation and vasopressors) Intervention Prediction

## Step-by-step instructions
### Requirments:
- psql (PostgreSQL 9.4 or higher)
- mimic code repo: 
- decompressed mimic dataset in MIMIC_DATA_DIR

### Create a conda env
``` bash
$ conda env create -f ./mimic_extract_env_py36.yml
```
### Build the database in PostgreSQL
In mimic code repo, navigte to the  mimic-iii/buildmimic/postgres/ directory and use make to run the Makefile. For example, to create MIMIC from a set of zipped CSV files in the "/path/to/data/" directory, run the following command:

``` bash
$ export datadir="MIMIC_DATA_DIR"
$ make create-user mimic

```
NOTE1: create_user, create a user with the default settings:
* Database name: `mimic`
* User name: `postgres`
* Password: `postgres`
* Schema: `mimiciii`
* Host: none (defaults to localhost)
* Port: none (defaults to 5432)

If you would like to change any of these parameters, you can either modify the makefile or do so in this make call:

``` bash
$ make create-user mimic datadir="/path/to/data/" DBNAME="my_db" DBPASS="my_pass" DBHOST="192.168.0.1"
```

NOTE2: mimic should run mimic-build and mimic-check. The first part takes a while.

NOTE3: If you have already set a pass for 'postgres' user, you might need to modify the defaults pass values in make files.


When using the database be sure to change the default search path to the mimic schema:

```bash
# connect to database mimic
$ psql -d mimic
# set default schema to mimiciii
mimic=# SET search_path TO mimiciii;

```
Ref of section: https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iii/buildmimic/postgres

### Building concepts
In mimic code repo, go to the concepts_postgres folder, run the postgres-functions.sql and postgres-make-concepts.sql scripts, in that order.
???????? activating the datavase and schemes?????????????
```bash
psql -U postgres -d mimic_test -f postgres-functions.sql
psql -U postgres -d mimic_test -f postgres-make-concepts.sql

```
If you face the follwoing error ":gzip: ccs_multi_dx.csv.gz: No such file or directory", navigate to concepts_postgres/diagnosis/ and run the sql file by itself:
```bash
psql -U postgres -d mimic_test -f ccs_dx.sql 
```

Next, you'll need to build 3 additional materialized views necessary for this pipeline. To do this (again with
schema edit permission), navigate to `utils` in MIMIC_EXTRACT: 
1. Run `bash postgres_make_extended_concepts.sh` after modifying the file as following:
- adding your username to CONNSTR : `export CONNSTR='-d mimic -U postgres'`
- defining MIMIC_CODE_DIR : `export MIMIC_CODE_DIR='/home/fazeleh/mimic-code/mimic-iii'`
- replacing `concepts` with `concepts_postgres` in both sql-file paths.

2. Run `psql -U postgres -d mimic -f niv-durations.sql`

### Set Cohort Selection and Extraction Criteria

Next, navigate to the root directory of _this repository_, activate your conda environment and run
`python mimic_direct_extract.py ...` with your args as desired.
python mimic_direct_extract.py --resource_path resources/ --out_path /home/fazeleh/physionet.org/files/mimiciii/1.4/data --psql_password 7272982 __psql_dbname mimic_test

#### Expected Outcome

The default setting will create an hdf5 file inside MIMIC_EXTRACT_OUTPUT_DIR with four tables:
* **patients**: static demographics, static outcomes
  * One row per (subj_id,hadm_id,icustay_id)

* **vitals_labs**: time-varying vitals and labs (hourly mean, count and standard deviation)
  * One row per (subj_id,hadm_id,icustay_id,hours_in)

* **vitals_labs_mean**: time-varying vitals and labs (hourly mean only)
  * One row per (subj_id,hadm_id,icustay_id,hours_in)

* **interventions**: hourly binary indicators for administered interventions
  * One row per (subj_id,hadm_id,icustay_id,hours_in)

Ref of this section: https://github.com/MLforHealth/MIMIC_Extract/blob/master/README.md#step-4-set-cohort-selection-and-extraction-criteria