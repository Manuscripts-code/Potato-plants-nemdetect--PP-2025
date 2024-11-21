#!/bin/bash

# Base paths
OUTPUT_DIR="./outputs"
MODEL="savgol-svc"
GROUP_ID=16

# Iterate over band-reduction values from 10 to 100 with a step of 10
for BAND_REDUCTION in {30..200..10}; do
  echo "Running with band-reduction=${BAND_REDUCTION}..."

  # Run the training and metrics generation commands
  pdm run main.py train-model $MODEL --do-optimize --group-id $GROUP_ID --band-reduction $BAND_REDUCTION &&
    pdm run main.py generate-metrics $MODEL --do-optimize --group-id $GROUP_ID --band-reduction $BAND_REDUCTION

  # Rename the generated metrics file
  METRICS_FILE="${OUTPUT_DIR}/${MODEL}__${GROUP_ID}__vnir-swir__1-2-3__True/results/metrics_reduced.txt"
  RENAMED_FILE="${OUTPUT_DIR}/${MODEL}__${GROUP_ID}__vnir-swir__1-2-3__True/results/metrics_reduced_${BAND_REDUCTION}.txt"

  if [ -f "$METRICS_FILE" ]; then
    mv "$METRICS_FILE" "$RENAMED_FILE"
    echo "Renamed metrics file to: $RENAMED_FILE"
  else
    echo "Metrics file not found for band-reduction=${BAND_REDUCTION}"
  fi
done
