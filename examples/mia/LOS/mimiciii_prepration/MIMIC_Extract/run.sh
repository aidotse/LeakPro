#!/bin/bash

# Config - equivalent to what you've provided in your Makefile
PHYSIONETURL="https://physionet.org/files/mimiciii/1.4/"
PHYSIONETDEMOURL="https://physionet.org/works/MIMICIIIClinicalDatabaseDemo/"

datadir="./data"  # Replace with actual datadir if needed

# Ensure that datadir ends in a slash
DATADIR=$(dirname "${datadir}")
if [[ "${datadir}" != "${DATADIR}/" ]]; then
  DATADIR="${DATADIR}/"
fi

# Set default parameters, can be overwritten via environment variables
DBNAME="${DBNAME:-mimic}"
DBUSER="${DBUSER:-postgres}"
DBPASS="${DBPASS:-postgres}"
DBSCHEMA="${DBSCHEMA:-mimiciii}"
DBHOST="${DBHOST:-}"
DBPORT="${DBPORT:-}"

# Construct DBSTRING
DBSTRING="dbname=${DBNAME} user=${DBUSER}"
if [ -n "$DBHOST" ]; then
  DBSTRING+=" host=${DBHOST}"
fi
if [ -n "$DBPORT" ]; then
  DBSTRING+=" port=${DBPORT}"
fi
if [ -n "$DBPASS" ]; then
  DBSTRING+=" password=${DBPASS}"
fi
DBSTRING+=" options=--search_path=${DBSCHEMA}"

# Part 1: Run the SQL file
echo '------------------------'
echo '-- Building MIMIC-III --'
echo '------------------------'
echo ''
sleep 2
echo 'Running niv-durations.sql'
psql -U "${DBUSER}" "${DBSTRING}" -f ./utils/niv-durations.sql

# Part 2: Run the Python script in the same directory

python3 mimic_direct_extract.py --resource_path resources/ --out_path output/ --psql_password postgres --psql_dbname mimic

# # Part 3: Copy the output to the data directory
# echo 'Copying file to target directory'
cp ./output/all_hourly_data.h5 ../../data/