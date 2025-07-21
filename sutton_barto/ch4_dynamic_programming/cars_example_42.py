from random import randint, random
from math import factorial, exp
from matplotlib import pyplot as plt

def poisson(lam, n):
    """Calculate Poisson probability mass function."""
    return (lam ** n) * (exp(-lam)) / factorial(n)

# Example 4.2, jack's rental

# 2 locations
# if rent, 10$ credit, unless no cars
# cars available after returning

# move car for cost 2$

# requests: rand poisson variables (n = lambda^n/n! exp(-lambda))
# Location A: lambda=3 for request, 3 for return
# Location B: lambda=4 for request, 2 for return

# no more than 20 cars per location (more disappear)

# discount rate 0.9

# days are steps,
# state is number of cars at each location at end of the day
# action is how many cars to move

# Initialization:
max_n = 20
state = (max_n, max_n)  # (cars at location A, cars at location B)
policy_state = [[0 for _ in range(max_n + 1)] for _ in range(max_n + 1)]

value_state = [[0 for _ in range(max_n + 1)] for _ in range(max_n + 1)]
print(value_state)


