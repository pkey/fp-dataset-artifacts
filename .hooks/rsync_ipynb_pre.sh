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

rsync -nrc --out-format="%n" "fp.ipynb" "$EVALUATED_PATH/fp.ipynb" > /tmp/rsync_diff.log


if [[ $? -ne 0 || -s /tmp/rsync_diff.log ]]; then
  echo "YOUR REMOTE FP.IPYNB HAS BEEN POTENTIALLY UPDATED!!!!"
  echo "This normally shouldn't happen. If you want to transfer your remote changes to local, use"
  echo "rsync -av '$EVALUATED_PATH/fp.ipynb' 'fp.ipynb'"
  exit 1
fi
