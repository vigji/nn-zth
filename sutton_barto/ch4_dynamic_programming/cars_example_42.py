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

lambdas_dict = {"locA": {"request": 4, "return": 3}, 
                "locB": {"request": 3, "return": 2}}

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

def trans_prob(state, action, location, max_n):
    request_lamb, return_lamb = lambdas_dict[location]["request"], lambdas_dict[location]["return"]

    probs_rent = [0 for _ in range(max_n + 1)]
    probs_returned = [0 for _ in range(max_n + 1)]
    # car requested:
    for n in range(state):
        probs_rent[n] = poisson(request_lamb, n)

    # car returned
    for n in range(max_n - state):
        probs_returned[n] = poisson(return_lamb, n)

    # final probabilities:
    for n in range(max_n + 1):
        

    return probs_rent, probs_returned


state = 10
arr, arr2 = trans_prob(state, 0, "locA", max_n)

plt.figure()
plt.plot(arr)
plt.plot(arr2)
plt.show()