#!/bin/bash
set -e


if [[ "$1" = "serve" ]]; then
    shift 1
    torchserve --start --ts-config /app/all_models.properties --disable-token-auth
else
    eval "$@"
fi

# prevent docker exit
tail -f /dev/null
