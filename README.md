# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with Grid World with Wind.

## PROBLEM STATEMENT
The agent is placed in a 5x5 grid where it must navigate to a goal position. However, in certain columns, wind affects the agent’s movement, pushing it upward. The challenge is to find an optimal policy that maximizes the cumulative reward while considering the stochastic effects of wind.
## POLICY EVALUATION FUNCTION
![image](https://github.com/user-attachments/assets/00b628f2-a4a8-4fbb-b0c2-09c9e5dac4a1)

## PROGRAM
```
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);
```
```
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")

```
```
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```
```
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
```
```
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
```
```
env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)
P

init_state

```
```
state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)
```
```
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,
    mean_return(env, pi_frozenlake)))
```
```
pi_2 = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: UP,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7: DOWN,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:UP,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]

```
```
print("Name: SANTHANA LAKSHMI K")
print("Register Number: 21222224009")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state) * 100,
    mean_return(env, pi_2)))
success_pi2 = probability_success(env, pi_2, goal_state=goal_state) * 100
mean_return_pi2 = mean_return(env, pi_2)
```
```
print("\nYour Policy Results:")
print(f"Reaches goal: {success_pi2:.2f}%")
print(f"Average undiscounted return: {mean_return_pi2:.4f}")
success_pi1 = probability_success(env, pi_frozenlake, goal_state=goal_state) * 100
mean_return_pi1 = mean_return(env, pi_frozenlake)
```
```
print("\nComparison of Policies:")
print(f"First Policy - Success Rate: {success_pi1:.2f}%, Mean Return: {mean_return_pi1:.4f}")
print(f"Your Policy  - Success Rate: {success_pi2:.2f}%, Mean Return: {mean_return_pi2:.4f}")
```
```
if success_pi1 > success_pi2:
    print("\nThe first policy is better based on success rate.")
elif success_pi2 > success_pi1:
    print("\nYour policy is better based on success rate.")
else:
    print("\nBoth policies have the same success rate.")

if mean_return_pi1 > mean_return_pi2:
    print("The first policy is better based on mean return.")
elif mean_return_pi2 > mean_return_pi1:
    print("Your policy is better based on mean return.")
else:
    print("Both policies have the same mean return.")
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[s] - v))
            V[s] = v

        if delta < theta:
            break

    return V
```
```

V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)

V2 = policy_evaluation(pi_2, P, gamma=0.99)

print("\nState-value function for Your Policy:")
print_state_value_function(V2, P, n_cols=4, prec=5)

if np.sum(V1 >= V2) == len(V1):
    print("\nThe first policy is the better policy.")
elif np.sum(V2 >= V1) == len(V2):
    print("\nYour policy is the better policy.")
else:
    print("\nBoth policies have their merits.")
V1>=V2
```
```
if(np.sum(V1>=V2)==11):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==11):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```


## OUTPUT:
Mention the first and second policies along with its state value function and compare them
![image](https://github.com/user-attachments/assets/feed2bea-01fa-4ceb-8118-ce6127147b5b)

![image](https://github.com/user-attachments/assets/36c717b2-abb6-4183-b9dd-303e6eb0c1ec)

![Screenshot 2025-03-18 230538](https://github.com/user-attachments/assets/60b86fe5-7602-462a-8dc9-3d0656c3b57b)

![Screenshot 2025-03-18 230553](https://github.com/user-attachments/assets/8dea55ed-93c7-4663-b125-b6a8d50257d6)

![Screenshot 2025-03-18 230603](https://github.com/user-attachments/assets/7dff1239-2507-4691-971c-9b42a4321e8d)

![Screenshot 2025-03-18 230603](https://github.com/user-attachments/assets/7dff1239-2507-4691-971c-9b42a4321e8d)

![Screenshot 2025-03-18 230613](https://github.com/user-attachments/assets/1c053afe-e535-4130-b9d6-064cc08bf4f2)

![Screenshot 2025-03-18 230628](https://github.com/user-attachments/assets/7e7437bd-7486-4725-9148-2e54955ec393)

![Screenshot 2025-03-18 230642](https://github.com/user-attachments/assets/395d9b7c-04cc-4b5a-9758-dd204ba4e921)

## RESULT:

Thus, the Given Policy has been Evaluated and Optimal Policy has been Computed using Python Programming and execcuted successfully.
