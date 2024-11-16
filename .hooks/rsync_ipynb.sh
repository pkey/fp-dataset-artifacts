#!/bin/bash

# Path to .env in the parent folder
ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    # Export variables directly from the .env file
    set -a  # Automatically export all variables
    source "$ENV_FILE"
    set +a  # Stop automatically exporting variables
    echo ".env file loaded from parent directory."
else
    echo ".env file not found in parent directory."
    exit 1
fi

EVALUATED_PATH=$(eval echo "$PATH_TO_COLAB_NOTEBOOK")
if [ ! -d "$EVALUATED_PATH" ]; then

    echo "Please point to your Colab Notebook in GDrive"
    exit 1
fi

rsync -av fp.ipynb "$EVALUATED_PATH/fp.ipynb"
