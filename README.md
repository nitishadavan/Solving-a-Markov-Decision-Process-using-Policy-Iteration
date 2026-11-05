# Solving a Markov Decision Process using Policy Iteration

## Aim

To implement the Policy Iteration algorithm for solving a finite Markov Decision Process using the Gymnasium FrozenLake-v1 environment, by repeatedly performing policy evaluation and policy improvement to obtain the optimal value function and optimal policy.

---

## Problem Statement

In this experiment, the `FrozenLake-v1` environment is solved using the **Policy Iteration** algorithm.

The agent starts from the start state and must reach the goal state without falling into holes. The environment is represented as a finite Markov Decision Process. Policy Iteration is used to repeatedly evaluate the current policy and improve it until the policy becomes stable.

The objective is to find:

1. The optimal state-value function $V^*(s)$
2. The optimal policy $pi^*(s)$

---

## Software Requirements

```bash
pip install gymnasium numpy
```

---

## Environment Description

The experiment uses the Gymnasium `FrozenLake-v1` environment.

FrozenLake is a grid-world environment where the agent moves over frozen tiles and tries to reach the goal without falling into holes.

For the default 4 × 4 FrozenLake map:

| Component | Description |
|---|---|
| Environment | `FrozenLake-v1` |
| Map size | 4 × 4 |
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching the goal, 0 otherwise |
| Terminal states | Goal and hole states |

---

## Theory

Policy Iteration is a Dynamic Programming method used to find the optimal policy of a Markov Decision Process.

It consists of two major steps:

1. **Policy Evaluation**
2. **Policy Improvement**

These two steps are repeated until the policy becomes stable.

---

## Policy Evaluation

Policy evaluation estimates the value function for the current policy.

The Bellman expectation equation is:

$$
V^\pi(s) =
\sum_a \pi(a \mid s)
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $pi(a \mid s)$ | Probability of taking action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $pi$ |

---

## Policy Improvement

Policy improvement updates the policy greedily with respect to the current value function.

The improved policy is obtained as:

$$
\pi'(s) =
\arg\max_a
\sum_{s'} P(s' \mid s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
$$

If the improved policy is the same as the old policy, the policy is considered stable.

---

## Algorithm

1. Create the Gymnasium `FrozenLake-v1` environment.
2. Initialize a random policy.
3. Repeat until the policy becomes stable:
   - Evaluate the current policy using iterative policy evaluation.
   - Improve the policy greedily using the current value function.
   - Compare the old policy and the new policy.
4. Stop when the policy does not change.
5. Display the optimal value function and optimal policy.

---

## Python Program

```python

# -------------------------------------------------
# Policy Evaluation
# -------------------------------------------------
policy = np.ones((n_states, n_actions)) / n_actions
def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    n_states = env.observation_space.n
    V = np.zeros(n_states)
    iterations = 0

    while True:
        delta = 0
        iterations += 1
        for s in range(n_states):
            v = V[s]
            new_value = 0
            for a in range(env.action_space.n):
                prob_a = policy[s][a]
                for prob, next_state, reward, done in env.P[s][a]:
                    new_value += prob_a * prob * (reward + gamma * V[next_state])
            V[s] = new_value
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V, iterations

V, iterations = policy_evaluation(env, policy, gamma, theta)

print("Name: Nitish Adavan D  ")
print("Register Number: 212224240107 ")
print("Number of iterations:", iterations)
print("\nState-Value Function:")
print(V)

print("\nName: Nitish Adavan D ")
print("Register Number: 212224240107 ")
print("\nState-Value Function as 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))


print("\nFinal Optimal Policy as 4x4 Grid:")

action_symbols = ['←', '↓', '→', '↑']

policy_grid = []

for s in range(n_states):

    if s in [5, 7, 11, 12]:
        policy_grid.append("H")

    elif s == 15:
        policy_grid.append("G")

    else:
        best_action = np.argmax(optimal_policy[s])
        policy_grid.append(action_symbols[best_action])

print(np.array(policy_grid).reshape(4, 4))



# -------------------------------------------------
# Policy Improvement
# -------------------------------------------------
def policy_improvement(env, V, gamma=0.99):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = np.zeros((n_states, n_actions))

    for state in range(n_states):
        action_values = np.zeros(n_actions)

        for action in range(n_actions):
            for prob, next_state, reward, terminated in env.P[state][action]:
                action_values[action] += prob * (
                    reward + gamma * V[next_state] * (not terminated)
                )

        best_action = np.argmax(action_values)
        policy[state, best_action] = 1.0

    return policy

#-------------------------------------------------
# Policy Iteration
# -------------------------------------------------
def policy_iteration(env, gamma=0.99, theta=1e-8):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    policy = np.ones((n_states, n_actions)) / n_actions

    while True:
        V, _ = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, V



```

## Output



### Total policy iterations:
<img width="1480" height="1062" alt="2710eef2-ad60-43f7-804e-fa5befbf147b" src="https://github.com/user-attachments/assets/f8da34a7-1d8a-4a8e-afb1-ab4409094149" />




### Optimal State-Value Function:
<img width="2172" height="724" alt="3aef0ab0-db2d-4bb5-936d-82aca25037c9" src="https://github.com/user-attachments/assets/97471352-9e65-4ddf-993e-d80bbe66a8e0" />



### Optimal Policy:
<img width="519" height="177" alt="image" src="https://github.com/user-attachments/assets/91ef06d8-6653-4d3c-99e9-2774debea116" />






## Result

The program displays:

  1.Number of policy iterations.

  2.Total policy evaluation iterations.

  3.Optimal state-value function.

  4.Optimal state-value function in a 4 × 4 grid.

  5.Optimal policy in a 4 × 4 grid.

  6.Optimal action probabilities for each state.

## Inference

The Policy Iteration algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The algorithm performed Policy Evaluation to calculate the state-value function and Policy Improvement to update the policy based on the maximum expected rewards. These steps were repeated until the policy became stable. Finally, the optimal state-value function and optimal policy were obtained, helping the agent identify the best actions to reach the goal while avoiding the holes in the FrozenLake environment.
