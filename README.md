# Naive Bayes Classifier

## Overview

This project is a Naive Bayes text classifier 
implemented in Python, designed to classify 
emails as spam or ham. The classifier uses the 
binary variant of Naive Bayesian Classification, 
which models a document as a vector of binary 
features indicating the presence or absence 
of specific words.

## System and Software Requirements

- Python 3.x
- pipenv
- Pandas library (installed with the pipenv install)

## Installation and Setup

### Setting Up Environment

#### Windows:
```sh
pip install pipenv
pipenv install
```

#### MacOS:
```sh
pip3 install pipenv
pipenv install
```

#### Interpreter:
Set the interpreter to pipenv so that the necessary packages work

## Usage

### Preparing the Dataset

Ensure that the email text files are placed in 
the corpus directory. 
Parts 1-9 should be used for training, 
and part 10 should be used for testing.

Replace the global variable 'directory' in the script
with the path to your 'corpus' folder including the parts:
```
directory = "/path/to/your/corpus"
```

### Running the Code

```sh
pipenv run python NBC_main.py
```

This will:
* Read and process the dataset.
* Build the vocabulary.
* Apply chi-quare feature selection.
* Train the Naive Bayesian Classifier.
* Evaluate the classifier.

### Output

Results will be printed in the console. 
The top 300 words with Chi-square values 
will be saved.

Without changes to the code, the 
output should match the attached_top_300_words.txt 
version.

## Contact

For any questions or issues, 
please contact Max Zeinstra at
m.e.g.zeinstra@student.utwente.nl.