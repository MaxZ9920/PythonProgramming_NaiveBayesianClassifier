import pandas as pd
import os
import string
from collections import defaultdict
import math

# Enter the location of the corpus folder here:
directory = "/path/to/your/corpus"


def build_dataset():
    """
    This function reads the dataset from the specified directory, splits it into training and test datasets,
    and returns them as pandas DataFrames.
    """
    # Initialize lists to hold the data for training, testing, and the complete dataset
    train_data, test_data, all_data = [], [], []

    # Loop through each folder in the directory
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)

        # Loop through each file in the folder
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            # Open and read the content of the file
            with open(file_path, 'r') as f:
                content = f.read()

                # Determine the label based on the file name
                label = "spam" if file.startswith("spm") else "ham"

                # Append the data to the appropriate list based on the folder name
                if folder == "part10":
                    # The last folder, part10, is used for test data
                    test_data.append({"category": label, "message": content})
                else:
                    # All other folders are used for training data
                    train_data.append({"category": label, "message": content})
                # Append the data to the complete dataset list
                all_data.append({"category": label, "message": content})

    # Convert the lists to pandas DataFrames and return the training, test, and complete datasets
    train = pd.DataFrame(train_data)
    test = pd.DataFrame(test_data)
    dataset = pd.DataFrame(all_data)
    return train, test, dataset


# Build the dataset
train_df, test_df, df = build_dataset()
print("Shape training data:", train_df.shape)
print("Shape test data:", test_df.shape)
print("Length full dataset:", len(df))
print('\n', train_df)


def text_to_words(text):
    """
    This function normalizes the text by converting it to lowercase, removing punctuation, removing numbers,
    and removing the word 'subject'. It returns a list of normalized words.
    """
    change_text = str.maketrans('', '', string.punctuation)
    text = text.lower().translate(change_text)
    text = ''.join([i for i in text if not i.isdigit()])
    text_to_words_list = text.split()
    text_to_words_list = [w for w in text_to_words_list if w != 'subject']
    return text_to_words_list


# Sample text normalization
sample_text = "Subject: I 12 just want to test if this works."
sample_words = text_to_words(sample_text)
print("\nSample text to words:", sample_words)


def build_vocabulary(dataframe):
    """
    This function builds a vocabulary from the training dataset, keeping track of the word counts in spam and ham messages.
    The vocabulary is returned as a dictionary where keys are words and values are lists of [spam count, ham count].
    """
    # Initialize a defaultdict to store the vocabulary with default values [0, 0]
    # Format is: {'word': [0, 1]} with [spam count, ham count]
    vocabulary = defaultdict(lambda: [0, 0])

    # Iterate through each row in the dataframe
    for index, row in dataframe.iterrows():
        # Convert the message to a set of tokens (words)
        tokens_set = set(text_to_words(row['message']))

        # Check the category of the message (spam or ham) and increment
        if row['category'] == 'spam':
            for token in tokens_set:
                vocabulary[token][0] += 1
        else:
            for token in tokens_set:
                vocabulary[token][1] += 1
    return vocabulary


# Build vocabulary of the training data
vocab = build_vocabulary(train_df)

# Print the first three rows of the vocab dictionary
vocab_items = list(vocab.items())[:3]
print("\nFirst three rows of vocab dict:")
for word, counts in vocab_items:
    print(f"Word: '{word}', occurs in {counts[0]} spam texts, occurs in {counts[1]} ham texts")


def text_count_to_tables(vocabulary, training_data):
    """
    This function creates tables for each word in the vocabulary, which are used to calculate chi-square values.
    """
    # Calculate the total number of texts
    N = len(training_data)
    N_spam = len(training_data[training_data['category'] == 'spam'])
    N_ham = len(training_data[training_data['category'] == 'ham'])

    print("\nLength total nr texts N:", N)
    print("Length N_spam: ", N_spam)
    print("Length N_ham: ", N_ham)

    tables_list = []

    for w, (spam_count, ham_count) in vocabulary.items():
        t = pd.DataFrame({
            'spam (c1)': [spam_count, N_spam - spam_count, N_spam],
            'ham (c2)': [ham_count, N_ham - ham_count, N_ham],
            'total': [spam_count + ham_count, N_spam - spam_count + N_ham - ham_count, N]
        }, index=[f'{w} (w)', f'not {w} (not w)', 'total (W, C)'])

        tables_list.append(t)

    return tables_list


# Create tables
tables = text_count_to_tables(vocab, train_df)

# Print the first three tables
print('\n')
for tb in tables[:3]:
    print(tb)
    print('\n')


def calculate_chi_square(tables_list):
    """
    This function calculates chi-square values for each word based on their tables.
    It returns a list of dictionaries containing the word, expected values, and chi-square value.
    """
    results = []

    for table in tables_list:
        w = table.index[0].split(' ')[0]

        # Extract observed values
        spam_word = table.loc[f'{w} (w)', 'spam (c1)']
        spam_not_word = table.loc[f'not {w} (not w)', 'spam (c1)']
        ham_word = table.loc[f'{w} (w)', 'ham (c2)']
        ham_not_word = table.loc[f'not {w} (not w)', 'ham (c2)']

        # Calculate totals
        W1 = spam_word + ham_word
        W2 = spam_not_word + ham_not_word
        C1 = spam_word + spam_not_word
        C2 = ham_word + ham_not_word
        N = table.loc['total (W, C)', 'total']

        # Calculate expected values
        E11 = (W1 * C1) / N
        E12 = (W1 * C2) / N
        E21 = (W2 * C1) / N
        E22 = (W2 * C2) / N

        # Calculate chi-square components
        # Division by 0 is not possible,
        # but is prevented by removing subject, which occurs in every text.
        chi2_11 = ((spam_word - E11) ** 2) / E11
        chi2_12 = ((ham_word - E12) ** 2) / E12
        chi2_21 = ((spam_not_word - E21) ** 2) / E21
        chi2_22 = ((ham_not_word - E22) ** 2) / E22

        # Sum chi-square components
        chi_square_value = chi2_11 + chi2_12 + chi2_21 + chi2_22

        # Store results
        results.append({
            'word': w,
            'expected_values': {
                'expected_spam_word': E11,
                'expected_ham_word': E12,
                'expected_spam_not_word': E21,
                'expected_ham_not_word': E22
            },
            'chi_square_value': chi_square_value,
            'components': f"{chi2_11:.2f} + {chi2_12:.2f} + {chi2_21:.2f} + {chi2_22:.2f}"
        })

    return results


def select_top_words(chi_square, n):
    """
    This function selects the top n words based on their chi-square values.
    It returns a list of dictionaries for the top n words.
    """
    # Sort the top n results based on chi_square_value in descending order
    sorted_results = sorted(chi_square, key=lambda dictionary: dictionary['chi_square_value'], reverse=True)
    # Select the top n words
    top_words = sorted_results[:n]
    return top_words


# Call chi square function
chi_square_results = calculate_chi_square(tables)

# Print examples of chi-square values
print("\nFirst three examples of chi-square values and expected values:")
for chi in chi_square_results[:3]:
    ev = chi['expected_values']
    print(f"Word: {chi['word']}")
    print(f"The expected values E(i, j) are: top row: "
          f"{ev['expected_spam_word']:.2f}, "
          f"{ev['expected_ham_word']:.2f}, bottom row: "
          f"{ev['expected_spam_not_word']:.2f}, "
          f"{ev['expected_ham_not_word']:.2f}")
    print(f"The x2 value for this table is {chi['chi_square_value']:.2f} ({chi['components']})\n")

# Select top 300, 200, and 100 words based on chi-square values
top_300_words = select_top_words(chi_square_results, 300)
top_200_words = select_top_words(chi_square_results, 200)
top_100_words = select_top_words(chi_square_results, 100)

# Print the top 3 words from the selections as example
print("Top 3 words from selection:")
for chi in top_300_words[:3]:
    print(f"Word: {chi['word']}, χ2 value: {chi['chi_square_value']:.2f}")


def computing_in_log_space(vocabulary, top_words, N_spam, N_ham):
    """
    This function calculates the log probabilities for the top words for spam and ham.
    It returns a dictionary where keys are words and values are their log probabilities.
    """
    # Initialize a dict for the probabilities
    w_probabilities = {}

    # Iterate through each word in the list of top words
    for w_info in top_words:
        w = w_info['word']
        # Extract the counts from the vocabulary
        spam_count, ham_count = vocabulary[w]
        # Calculate the smoothed probabilities
        P_w_spam = (spam_count + 1) / (N_spam + 2)
        P_w_ham = (ham_count + 1) / (N_ham + 2)

        # Calculate the log probabilities and store them
        w_probabilities[w] = {
            'P(w|spam)': math.log2(P_w_spam),
            'P(w|ham)': math.log2(P_w_ham),
            'P(~w|spam)': math.log2(1 - P_w_spam),
            'P(~w|ham)': math.log2(1 - P_w_ham)
        }
    return w_probabilities


# Calculate the number of spam and ham messages in the training dataset
N_spam_train = len(train_df[train_df['category'] == 'spam'])
N_ham_train = len(train_df[train_df['category'] == 'ham'])

# Compute log probabilities for the top words
word_probabilities_300 = computing_in_log_space(vocab, top_300_words, N_spam_train, N_ham_train)
word_probabilities_200 = computing_in_log_space(vocab, top_200_words, N_spam_train, N_ham_train)
word_probabilities_100 = computing_in_log_space(vocab, top_100_words, N_spam_train, N_ham_train)


def classify_text(text, w_probabilities, N_spam, N_ham):
    """
    This function classifies a given text as spam or ham based on the log probabilities.
    """
    tokens = set(text_to_words(text))

    # Calculate the initial log probabilities
    log_P_spam = math.log2(N_spam / N_spam + N_ham)
    log_P_ham = math.log2(N_ham / N_spam + N_ham)

    # Iterate through each word in the probabilities dictionary
    for w, probs in w_probabilities.items():
        if w in tokens:
            # If the word is in the tokens, add the log probability of the word given spam and ham
            log_P_spam += probs['P(w|spam)']
            log_P_ham += probs['P(w|ham)']
        else:
            # If the word is not in the tokens, add the log probability of the word not given spam and ham
            log_P_spam += probs['P(~w|spam)']
            log_P_ham += probs['P(~w|ham)']

    # Return the classification based on the higher log probability
    return 'spam' if log_P_spam > log_P_ham else 'ham'


def evaluate_classifier(testdata, w_probabilities, N_spam, N_ham):
    """
    This function evaluates the classifier on the test dataset and returns the accuracy.
    """
    # Initialize count for correct predictions
    correct_predictions = 0

    # Iterate over the testdata
    for index, row in testdata.iterrows():
        prediction = classify_text(row['message'], w_probabilities, N_spam, N_ham)
        if prediction == row['category']:
            correct_predictions += 1
    accuracy = (correct_predictions / len(testdata)) * 100
    return accuracy


# Calculate the number of spam and ham messages in the test dataset
N_spam_test = len(test_df[test_df['category'] == 'spam'])
N_ham_test = len(test_df[test_df['category'] == 'ham'])

# Evaluate the classifier with different sets of top words
accuracy_300 = evaluate_classifier(test_df, word_probabilities_300, N_spam_test, N_ham_test)
print(f"\nAccuracy with top 300 words: {accuracy_300:.2f} %")

accuracy_200 = evaluate_classifier(test_df, word_probabilities_200, N_spam_test, N_ham_test)
print(f"Accuracy with top 200 words: {accuracy_200:.2f} %")

accuracy_100 = evaluate_classifier(test_df, word_probabilities_100, N_spam_test, N_ham_test)
print(f"Accuracy with top 100 words: {accuracy_100:.2f} %")

# Save the top 300 words with their chi-square values to a text file
with open("attached_top_300_words.txt", "w") as tf:
    for chi in top_300_words:
        tf.write(f"{chi['word']}: {chi['chi_square_value']:.2f}\n")
