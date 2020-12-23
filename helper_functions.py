from constants import *


def debug(txt):
    """
    Print first level debug message
    """
    print(f'DEBUG: {txt}...', end='') if DEBUG else None


def debug2(txt):
    """
    Print second level debug message
    """
    print(f'DEBUG: {txt}') if DEBUG2 else None


def done():
    """
    Print first level done message
    """
    print("DONE") if DEBUG else None


def done2():
    """
    Print second level done message
    """
    print("DONE") if DEBUG2 else None


def error(txt):
    print(f'ERROR: {txt}')

def info(txt):
    print(f'INFO:  {txt}')


def print_metrics(list):
    print("--- METRICS ---------------------------------------------------")
    print(f'Accuracy: {list[ACCURACY]}')
    print(f'Precision: {list[PRECISION]}')
    print(f'Recall: {list[RECALL]}')
    print(f'Specificity: {list[SPECIFICITY]}')
    print("---------------------------------------------------------------")
