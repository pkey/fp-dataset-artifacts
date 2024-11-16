#!/bin/bash

# Path to .env in the parent folder
ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
    # Export variables directly from the .env file
    set -a  # Automatically export all variables
    source "$ENV_FILE"
    set +a  # Stop automatically exporting variables
else
    echo ".env file not found in parent directory."
    exit 1
fi

EVALUATED_PATH=$(eval echo "$PATH_TO_COLAB_NOTEBOOK")
if [ ! -d "$EVALUATED_PATH" ]; then

    echo "Please point to your Colab Notebook in GDrive"
    exit 1
fi

echo "Synchronising local fp.ipynb to the one in GDrive... Any changes in the cloud will be overwritten."

rsync -av fp.ipynb "$EVALUATED_PATH/fp.ipynb"
