import json
import requests
import random
import string
import secrets
import time
import re
import collections
#
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
#


try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode


from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



vowels = ['a', 'e', 'i', 'o', 'u']


# for ratio of vowels to length of word
def vowel_count(clean_word):
    count = 0
    for i in clean_word:
        if i in vowels:
            count = count+1.0
    return count/len(clean_word)


f = open("Words train 250000.txt")
df = []
for x in f:
  df.append(x)



for i in range(len(df)):
    df[i] = df[i].replace("\n", "")


l=[]
for words in df:
    l.append(vowel_count(words))
l = pd.Series(l)
l.describe()



bins = np.arange(0.0, 1.0, 0.05)
plt.hist(x = l, bins = bins)


max_length = 0
for words in df:
    if(len(words)>max_length):
        max_length = len(words)

n_word_dictionary = {i:[] for i in range(3, 30)}
count = 3
while count<=max_length:
    for words in df:
        if(len(words)>=count):
            for i in range(len(words)-count+1):
                #if words[i:i+count-1] not in n_word_dictionary[count]:
                n_word_dictionary[count].append(words[i:i+count])
    count = count+1


# function to find number of times a letter come in whole dictionary, keeping count of letter 1 if it comes in a word else 0
def func(new_dictionary):
    dictx = collections.Counter()
    for words in new_dictionary:
        temp = collections.Counter(words)
        for i in temp:
            temp[i] = 1
            dictx = dictx + temp
    return dictx


# function to generate a list of words which are substring in the original dictionary and of same length as clean word
def func2(n_word_dictionary, clean_word):
    new_dictionary = []
    l = len(clean_word)
    for dict_word in n_word_dictionary[l]:
        if re.match(clean_word,dict_word):
            new_dictionary.append(dict_word)
    return new_dictionary


HANGMAN_URL = "https://www.trexsim.com/trexsim/hangman"

class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []

        # Define character mappings before initializing the model
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(string.ascii_lowercase)}  # +1 for padding token at index 0
        self.index_to_char = {idx: char for char, idx in self.char_to_index.items()}

        # Load the dictionary and initialize the model
        self.full_dictionary = self.build_dictionary("Words train 250000.txt")
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        self.current_dictionary = self.full_dictionary.copy()  # Start with the full dictionary
        self.model = self.build_lstm_model(len(self.char_to_index) + 1, max(len(word) for word in self.full_dictionary))

    def encode_word(self, word):
        # Converts a word to a sequence of indexes
        encoded = [self.char_to_index.get(char, 0) for char in word]
        return pad_sequences([encoded], maxlen=self.max_length, padding='post')

    def build_lstm_model(self, vocab_size, max_length):
        model = Sequential([
            Embedding(vocab_size, 64, input_length=max_length),
            LSTM(50),
            Dense(vocab_size, activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def guess(self, word):
        clean_word = word[::2].replace("_", ".")
        len_word = len(clean_word)
        self.current_dictionary = [w for w in self.current_dictionary if len(w) == len_word and re.match(clean_word.replace('.', '[a-z]'), w)]
        
        # Use heuristic if less than 50% of letters are visible
        visible_letters = clean_word.count('.') / len(clean_word)
        if visible_letters > 0.5:
            return self.heuristic_guess(clean_word)
        
        # Use LSTM for prediction otherwise
        encoded_input = self.encode_word(clean_word.replace('.', ' '))
        prediction = self.model.predict(encoded_input)[0]
        max_prob = np.max(prediction)
        guess_idx = np.argmax(prediction)
        guess_letter = self.index_to_char.get(guess_idx, None)

        # Fallback to heuristic if low confidence or invalid guess
        if max_prob < 0.05 or not guess_letter or guess_letter in self.guessed_letters:
            guess_letter = self.heuristic_guess(clean_word)

        if guess_letter not in self.guessed_letters:
            self.guessed_letters.append(guess_letter)
        else:
            # If all letters were guessed but still called, pick any non-guessed letter
            for letter in string.ascii_lowercase:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break

        return guess_letter

    def heuristic_guess(self, pattern):
        # Calculate the frequency of each letter in the current dictionary
        letter_counts = collections.Counter(c for word in self.current_dictionary for c in word if c not in self.guessed_letters)
        if letter_counts:
            return max(letter_counts, key=letter_counts.get)
        return 'e'  # Fallback to most common letter if no suitable letter is found












    ##########################################################
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
                         
        response = self.request("/new_game", {"practice":practice})
        if response.get('status')=="approved":
            game_id = response.get('game_id')
            word = response.get('word')
            tries_remains = response.get('tries_remains')
            if verbose:
                print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
            while tries_remains>0:
                # get guessed letter from user code
                guess_letter = self.guess(word)
                    
                # append guessed letter to guessed letters field in hangman object
                self.guessed_letters.append(guess_letter)
                if verbose:
                    print("Guessing letter: {0}".format(guess_letter))
                    
                try:    
                    res = self.request("/guess_letter", {"request":"guess_letter", "game_id":game_id, "letter":guess_letter})
                except HangmanAPIError:
                    print('HangmanAPIError exception caught on request.')
                    continue
                except Exception as e:
                    print('Other exception caught on request.')
                    raise e
               
                if verbose:
                    print("Sever response: {0}".format(res))
                status = res.get('status')
                tries_remains = res.get('tries_remains')
                if status=="success":
                    if verbose:
                        print("Successfully finished game: {0}".format(game_id))
                    return True
                elif status=="failed":
                    reason = res.get('reason', '# of tries exceeded!')
                    if verbose:
                        print("Failed game: {0}. Because of: {1}".format(game_id, reason))
                    return False
                elif status=="ongoing":
                    word = res.get('word')
        else:
            if verbose:
                print("Failed to start a new game")
        return status=="success"
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        try:
            # response = self.session.request(
            response = requests.request(
                method or "GET",
                HANGMAN_URL + path,
                timeout=self.timeout,
                params=args,
                data=post_args)
        except requests.HTTPError as e:
            response = json.loads(e.read())
            raise HangmanAPIError(response)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)


api = HangmanAPI(access_token="a1b7350fec09520b223b99eb36949c", timeout=2000)


api.start_game(practice=1,verbose=True)
[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() #Get my game stats: (# of tries, # of wins)
practice_success_rate = total_practice_successes / total_practice_runs
print('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))

