#!/bin/bash
export SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
exec /usr/local/searxng/searx-pyenv/bin/python -m searx.webapp

