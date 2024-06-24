"""
Data preprocessor to be used while running the rationalizer
"""

import pandas as pd
import re

def remove_space(text):
    return re.sub(' +', ' ', text)

def replace_newline(text):
    text = text.replace('<\\newline> <\\newline>', '')
    text = text.replace('<\\newline> <\\newline> <\\newline>', '')
    text = text.replace('<\\newline> <\\newline> <\\newline> <\\newline>', '')
    return text

# replace some tokens
def process_story(text):
    text = text.replace('\t', '')
    text = text.replace('\n', '')
    text = text.replace('<br>', '')
    text = remove_space(text)
    text = replace_newline(text)
    return text

