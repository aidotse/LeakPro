# Config
PHYSIONETURL=https://physionet.org/files/mimiciii/1.4/
PHYSIONETDEMOURL=https://physionet.org/works/MIMICIIIClinicalDatabaseDemo/

# Ensure that datadir ends in a slash
DATADIR := $(dir $(datadir))
ifneq ($(datadir),$(DATADIR))
DATADIR := $(datadir)/
endif


# Set the following parameters to defaults
# These will be overwritten by settings passed to the makefile
DBNAME := mimic
DBUSER := postgres
DBPASS := postgres
DBSCHEMA := mimiciii

# NOTE: you do not need to specify localhost/port
# in fact, this is detrimental if you want to use peer authentication
# "localhost" uses a loopback, so peer authentication doesn't work with it
DBHOST :=
DBPORT :=

# when connecting, we use a single variable: DBSTRING
# **do not modify this**
# the create_mimic_user.sh script uses the above variables to create the DB/users
# so it's important to keep this DBSTRING consistent
DBSTRING := dbname=$(DBNAME) user=$(DBUSER)
ifneq ($(DBHOST),)
DBSTRING +=  host=$(DBHOST)
endif
ifneq ($(DBPORT),)
DBSTRING +=  port=$(DBPORT)
endif
ifneq ($(DBPASS),)
DBSTRING +=  password=$(DBPASS)
endif
# add ON_ERROR_STOP so scripts stop when they fail
DBSTRING +=  options=--search_path=$(DBSCHEMA)

# if psql version is greater than 10, then we can use declarative partitioning
# this is done in the postgres_create_tables_pg10.sql file
PSQL_SERVER_VERSION := $(shell echo SHOW SERVER_VERSION_NUM | psql "$(DBSTRING)" -At)
PSQL_SERVER_GREATER_THAN_10 := $(shell /usr/bin/test $(PSQL_SERVER_VERSION) -ge 100000 ; echo $$?)
ifeq ($(PSQL_SERVER_GREATER_THAN_10),0)
        PSQL_MAJOR_VERSION := _pg10
endif

help:
	@echo 'Usage: make <target>'
	@echo ''
	@echo 'Targets:'
	@echo '  mimic-download: Download MIMIC-III from PhysioNet'
	@echo '  mimic-check: Check that the downloaded data is present'
	@echo '  mimic-build: Build the MIMIC-III database'
	@echo '  mimic-builder: Build the MIMIC-III database'
	@echo '  mimic-build-gz: Build the MIMIC-III database'
	@echo '  mimic_download_check: Download and check the MIMIC-III data'
	@echo '  create-user: Create the MIMIC-III user, database, and schema'
	@echo '  concepts: Add the MIMIC-III concepts to the database'
	@echo '  extra_concept: Add the MIMIC-III extra concepts to the database'
	@echo '  clean: Clean up the build directory'
	@echo ''
	@echo 'Variables:'
	@echo '  datadir: The directory to download the MIMIC-III data to'
	@echo '  physionetuser: The PhysioNet username to use for downloading the data'
	@echo '  DBNAME: The name of the database to create'
	@echo '  DBUSER: The name of the user to create'
	@echo '  DBPASS: The password for the user'
	@echo '  DBSCHEMA: The schema to create'
	@echo '  DBHOST: The host to connect to'
	@echo '  DBPORT: The port to connect to'
	@echo ''
	@echo 'Example:'
	@echo '  make mimic-download datadir=/path/to/data
mimic-build:create-user mimic-builder concepts extra_concept
mimic_download_check:  mimic-download mimic-check-csv

create-user:
	@echo '-------------------------------------'
	@echo '-- Creating user, database, schema --'
	@echo '-------------------------------------'
	@echo ''
	@sleep 2
	MIMIC_USER="$(DBUSER)" MIMIC_DB="$(DBNAME)" MIMIC_PASSWORD="$(DBPASS)" MIMIC_SCHEMA="$(DBSCHEMA)" DBHOST="$(DBHOST)" DBPORT="$(DBPORT)" ./create_mimic_user.sh

mimic-build-gz:
	@echo '------------------------'
	@echo '-- Building MIMIC-III --'
	@echo '------------------------'
	@echo ''
	@echo '---------------------'
	@echo '-- Creating schema --'
	@echo '---------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -c "DROP SCHEMA IF EXISTS $(DBSCHEMA); CREATE SCHEMA $(DBSCHEMA);"
	@echo ''
	@echo '---------------------'
	@echo '-- Creating tables --'
	@echo '---------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_create_tables$(PSQL_MAJOR_VERSION).sql
	@echo ''
	@echo '------------------'
	@echo '-- Loading data --'
	@echo '------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_load_data_gz.sql -v mimic_data_dir=${datadir}
	@echo ''
	@echo '--------------------'
	@echo '-- Adding indexes --'
	@echo '--------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_add_indexes.sql
	@echo ''
	@echo '--------------------'
	@echo '-- Adding comments --'
	@echo '--------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_add_comments.sql
	@echo ''
	@echo '----------------'
	@echo '-- Check data --'
	@echo '----------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_checks.sql

mimic-builder:
	@echo '------------------------'
	@echo '-- Building MIMIC-III --'
	@echo '------------------------'
	@echo ''
	@echo '---------------------'
	@echo '-- Creating schema --'
	@echo '---------------------'
	@echo ''
	@sleep 2
	@echo "DBSCHEMA is: $(DBSCHEMA)"
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -c "DROP SCHEMA IF EXISTS $(DBSCHEMA); CREATE SCHEMA $(DBSCHEMA);"
	@echo ''
	@echo '---------------------'
	@echo '-- Creating tables --'
	@echo '---------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_create_tables$(PSQL_MAJOR_VERSION).sql
	@echo ''
	@echo '------------------'
	@echo '-- Loading data --'
	@echo '------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_load_data.sql -v mimic_data_dir=${datadir}
	@echo ''
	@echo '--------------------'
	@echo '-- Adding indexes --'
	@echo '--------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_add_indexes.sql
	@echo ''
	@echo '--------------------'
	@echo '-- Adding comments --'
	@echo '--------------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_add_comments.sql
	@echo ''
	@echo '----------------'
	@echo '-- Check data --'
	@echo '----------------'
	@echo ''
	@sleep 2
	psql "$(DBSTRING)" -v ON_ERROR_STOP=1 -f postgres_checks.sql

mimic-download:
	@echo '------------------------------------------'
	@echo '-- Downloading MIMIC-III from PhysioNet --'
	@echo '------------------------------------------'
	@echo ''
	@echo "DATADIR is $(DATADIR)"cd 
ifeq ("$(physionetuser)","")
	@echo 'Cannot download data: physionetuser variable is unset.'
	@echo 'Call the makefile again with physionetuser=<USERNAME>'
	@echo ' e.g. make eicu-download datadir=/path/to/data physionetuser=hello@physionet.org'
else
	wget --user $(physionetuser) --ask-password -P $(DATADIR) -A *.csv.gz -m -p -E -k -K -np -nd "$(PHYSIONETURL)"
	@echo '------------------------------------------'
	@echo '-- Unzipping downloaded files --'
	@echo '------------------------------------------'
	gunzip $(DATADIR)/*.csv.gz
endif



#This is fairly inelegant and could be tidied with a for loop and an if to check for gzip,
#but need to maintain compatibility with Windows, which baffling lacks these things
mimic-check-gz:
	@echo ''
	@echo '-----------------------'
	@echo '-- Checking for data --'
	@echo '-----------------------'
	@echo ''
	@echo 'Data path: ' $(DATADIR)
ifeq ("$(wildcard $(DATADIR)ADMISSIONS.csv.gz)","")
	@echo "Unable to find $(DATADIR)ADMISSIONS.csv.gz - exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)CALLOUT.csv.gz)","")
	@echo "Unable to find $(DATADIR)CALLOUT.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)CAREGIVERS.csv.gz)","")
	@echo "Unable to find $(DATADIR)CAREGIVERS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)CHARTEVENTS.csv.gz)","")
	@echo "Unable to find $(DATADIR)CHARTEVENTS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)CPTEVENTS.csv.gz)","")
	@echo "Unable to find $(DATADIR)CPTEVENTS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)DATETIMEEVENTS.csv.gz)","")
	@echo "Unable to find $(DATADIR)DATETIMEEVENTS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)D_CPT.csv.gz)","")
	@echo "Unable to find $(DATADIR)D_CPT.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)DIAGNOSES_ICD.csv.gz)","")
	@echo "Unable to find $(DATADIR)DIAGNOSES_ICD.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)D_ICD_DIAGNOSES.csv.gz)","")
	@echo "Unable to find $(DATADIR)D_ICD_DIAGNOSES.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)D_ICD_PROCEDURES.csv.gz)","")
	@echo "Unable to find $(DATADIR)D_ICD_PROCEDURES.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)D_ITEMS.csv.gz)","")
	@echo "Unable to find $(DATADIR)D_ITEMS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)D_LABITEMS.csv.gz)","")
	@echo "Unable to find $(DATADIR)D_LABITEMS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)DRGCODES.csv.gz)","")
	@echo "Unable to find $(DATADIR)DRGCODES.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)ICUSTAYS.csv.gz)","")
	@echo "Unable to find $(DATADIR)ICUSTAYS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)INPUTEVENTS_CV.csv.gz)","")
	@echo "Unable to find $(DATADIR)INPUTEVENTS_CV.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)INPUTEVENTS_MV.csv.gz)","")
	@echo "Unable to find $(DATADIR)INPUTEVENTS_MV.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)LABEVENTS.csv.gz)","")
	@echo "Unable to find $(DATADIR)LABEVENTS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)MICROBIOLOGYEVENTS.csv.gz)","")
	@echo "Unable to find $(DATADIR)MICROBIOLOGYEVENTS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)NOTEEVENTS.csv.gz)","")
	@echo "Unable to find $(DATADIR)NOTEEVENTS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)OUTPUTEVENTS.csv.gz)","")
	@echo "Unable to find $(DATADIR)OUTPUTEVENTS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)PATIENTS.csv.gz)","")
	@echo "Unable to find $(DATADIR)PATIENTS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)PRESCRIPTIONS.csv.gz)","")
	@echo "Unable to find $(DATADIR)PRESCRIPTIONS.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)PROCEDUREEVENTS_MV.csv.gz)","")
	@echo "Unable to find $(DATADIR)PROCEDUREEVENTS_MV.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)PROCEDURES_ICD.csv.gz)","")
	@echo "Unable to find $(DATADIR)PROCEDURES_ICD.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)SERVICES.csv.gz)","")
	@echo "Unable to find $(DATADIR)SERVICES.csv.gz. Exiting before build."; exit 1
endif
ifeq ("$(wildcard $(DATADIR)TRANSFERS.csv.gz)","")
	@echo "Unable to find $(DATADIR)TRANSFERS.csv.gz. Exiting before build."; exit 1
endif
	@echo 'All data present!'
	@echo ''

clean:
	@echo

concepts:
	@echo '------------------------'
	@echo '-- Building MIMIC-III --'
	@echo '------------------------'
	@echo ''
	@echo '---------------------'
	@echo '-- Adding concepts --'
	@echo '---------------------'
	@echo ''
	@sleep 2
	@echo "DBSCHEMA is: $(DBSCHEMA)"
	(cd ../concepts_postgres/ && \
	psql "dbname=mimic user=postgres password=postgres options=--search_path=$(DBSCHEMA)" -v ON_ERROR_STOP=1 -f postgres-functions.sql && \
	psql "dbname=mimic user=postgres password=postgres options=--search_path=$(DBSCHEMA)" -v ON_ERROR_STOP=1 -f postgres-make-concepts.sql)
	cd ../concepts_postgres/diagnosis/ && \
	psql "dbname=mimic user=postgres password=postgres options=--search_path=$(DBSCHEMA)" -v ON_ERROR_STOP=1 -f ccs_dx.sql




extra_concept:
	@echo '------------------------'
	@echo '-- Building MIMIC-III --'
	@echo '------------------------'
	@echo ''
	@sleep 2
	@echo 'Running niv-durations.sql'
	psql -U ${DBUSER} "${DBSTRING}" -f ../../MIMIC_Extract/utils/niv-durations.sql



.PHONY: help mimic clean