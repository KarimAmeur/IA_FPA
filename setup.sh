#!/bin/bash

# Installer les dépendances système requises
apt-get update
apt-get install -y \
    libgl1 \
    libsm6 \
    libxrender1 \
    libxext6