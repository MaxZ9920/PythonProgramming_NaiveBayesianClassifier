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
- Pandas library
- A text editor or IDE (e.g., PyCharm)

## Installation and Setup

### Cloning the Repository and Setting Up Environment
```sh
git clone <repository_url>
cd PythonProgramming_assignment_NBC
pip install pipenv
pipenv install
```

## Usage

### Preparing the Dataset

Ensure that the email text files are placed in 
the corpus directory. 
Parts 1-9 should be used for training, 
and part 10 should be used for testing.

Replace the global variable 'directory' in the script
with the path to your 'corpus' folder including the parts.

### Running the Code

```sh
pipenv run python src/NBC_main.py
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