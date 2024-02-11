# Image recognition using FHE

## Run the application on your machine

In this directory, ie `fhe`, you can do the following steps.

### Install dependencies

First, position yourself in the root directory of the project and run the following `make` command:

```bash
cd ..
make deps
```

Then, install git LFS to get both the server and client used to run the webapp:

```bash
sudo apt-get install git-lfs
# or for macos: brew install git-lfs
git lfs install
git lfs pull
```

## Run the app 

In a terminal, run:

```bash
make webapp
```

## Interact with the application

Open the given URL link that defaults to port `8888` (search in your terminal for a line `Running on local URL:  http://127.0.0.1:8888/` or [press here](http://localhost:8888/) to open the default port link).

