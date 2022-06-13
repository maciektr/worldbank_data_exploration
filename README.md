# World Bank Data Exploration
## Setup
This project requires both [python] and [R] setup to run. 
We recommend using [pyenv](https://github.com/pyenv/pyenv) or similar tool for python version management.

To create runtime environment and install required dependencies, run the following script. 
```bash
git clone https://github.com/maciektr/worldbank_data_exploration.git
cd worldbank_data_exploration
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
echo "install.packages('TSdist')" | R --no-save
```

## Usage
### Dataset
To download the dataset, run following command. Alternatively, you can use packaged indicators from `indicators.tar.gz` 
archive by unpacking and placing them in the `data/indicators` folder. 
```bash
python -m data_sources.load_dataset
```

## Development
### Code formatting 
You can format the code automatically with [black](https://github.com/psf/black) by running the following command.
```bash
black .
```

[python]: https://www.python.org/
[R]: https://www.r-project.org/
[TSdist]: https://cran.r-project.org/web/packages/TSdist/index.html
