# Q-Learning Formula

## The Q-Learning Update Rule

The core Q-learning algorithm updates Q-values using the following formula:

$$Q(s,a) = Q(s,a) + \alpha[r + \gamma \max Q(s',a') - Q(s,a)]$$

## Formula Components

- **$Q(s,a)$**: The Q-value for state $s$ and action $a$
- **$\alpha$**: Learning rate ($0 < \alpha \leq 1$)
  - Controls how much new information overrides old information
  - Higher values mean faster learning but less stability
- **$r$**: Immediate reward received after taking action $a$ in state $s$
- **$\gamma$**: Discount factor ($0 \leq \gamma \leq 1$)
  - Determines importance of future rewards
  - $\gamma = 0$: Only immediate rewards matter
  - $\gamma = 1$: Future rewards are as important as immediate rewards
- **$\max Q(s',a')$**: Maximum Q-value for the next state $s'$ over all possible actions $a'$
- **$s'$**: Next state after taking action $a$ in state $s$

## Temporal Difference Error

The term $[r + \gamma \max Q(s',a') - Q(s,a)]$ is called the **temporal difference error**:
- It represents the difference between the expected reward and the current Q-value estimate
- When positive: Current estimate is too low
- When negative: Current estimate is too high

## Algorithm Properties

- **Model-free**: Doesn't require knowledge of environment dynamics
- **Off-policy**: Can learn optimal policy while following exploratory policy
- **Convergence**: Guaranteed to converge to optimal Q-values under certain conditions