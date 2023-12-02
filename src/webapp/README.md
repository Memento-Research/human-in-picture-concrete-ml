---
title: Image Filtering On Encrypted Data Using Fully Homomorphic Encryption
emoji: 📸 🌄
colorFrom: purple
colorTo: purple
sdk: gradio
sdk_version: 3.40.1
app_file: app.py
pinned: true
tags: [FHE, PPML, privacy, privacy preserving machine learning, image processing, 
  homomorphic encryption, security]
python_version: 3.8.16
---

# Image filtering using FHE

## Run the application on your machine

In this directory, ie `image_filtering`, you can do the following steps.

### Install dependencies

First, create a virtual env and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then, install required packages:

```bash
pip3 install pip --upgrade
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```

The above steps should only be done once.

## Run the app 

In a terminal, run:

```bash
source .venv/bin/activate
python3 app.py
```

## Interact with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/`).


## Generate new filters

It is also possible to manually add some new filters in `filters.py`. Yet, in order to be able to use
them interactively in the app, you first need to update the `AVAILABLE_FILTERS` list found in `common.py`
and then compile them by running :

```bash
python3 generate_dev_filters.py
```

Check it finishes well (by printing "Done!").

