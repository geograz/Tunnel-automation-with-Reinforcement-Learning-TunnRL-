# -*- coding: utf-8 -*-
"""
test code for Github repo
"""


epsilon = 1  # initial exploration
MIN_EPSILON = 0.05  # final exploration
EPSILON_DECAY = 0.99997  # Every episode will be epsilon*EPS_DECAY

step = 1

while epsilon > MIN_EPSILON:
    epsilon *= EPSILON_DECAY
    step += 1

print(f'it wil take {step} episodes to reach the minimum epsilon')

def print_function(some_text):
    """a very complicated function"""
    print(some_text)
