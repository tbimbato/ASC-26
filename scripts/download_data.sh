#!/bin/sh
# Riscarica in data/raw/ i 3 dataset reali di RIR usati da src/real_test.py.
# Da lanciare dalla root della repo: sh scripts/download_data.sh

set -e
RAW=data/raw
ACE_FOLDER="$RAW/ACE[Data Corpus for the IEEE-AASP Challenge on the Acoustic Characterization of Environments (ACE)]"
mkdir -p "$RAW" "$ACE_FOLDER"

# BUT ReverbDB, solo RIR (9.3 GB da scaricare, 16 GB estratto)
curl -L -C - --fail -o "$RAW/BUT_ReverbDB_rel_19_06_RIR-Only.tgz" \
  "https://merlin.fit.vutbr.cz/ReverbDB/BUT_ReverbDB_rel_19_06_RIR-Only.tgz"
tar -xzf "$RAW/BUT_ReverbDB_rel_19_06_RIR-Only.tgz" -C "$RAW"

# ACE corpus, solo la parte Single (0.4 GB), Zenodo 6257551
curl -L -C - --fail -o "$RAW/ACE_Corpus_RIRN_Single.tbz2" \
  "https://zenodo.org/api/records/6257551/files/ACE_Corpus_RIRN_Single.tbz2/content"
tar -xjf "$RAW/ACE_Corpus_RIRN_Single.tbz2" -C "$ACE_FOLDER"

# AIR 1.4 (0.2 GB), mirror OpenSLR 20
# ATTENZIONE: lo zip contiene file .mat, non .wav. La cartella AIR_wav_files
# era stata convertita a parte (vedi load_air.m dentro lo zip).
curl -L -C - --fail -o "$RAW/air_database_release_1_4.zip" \
  "https://openslr.trmal.net/resources/20/air_database_release_1_4.zip"
unzip -q -o "$RAW/air_database_release_1_4.zip" -d "$RAW"
