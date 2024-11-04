import numpy as np
import random
from env import GridEnvDeform

def eval_tabular(env : GridEnvDeform, Q,state_dict, num_episodes=100, max_episode_steps=100):
    total_rewards = []

    for episode in range(num_episodes):
        s, _ = env.reset()
        state = state_dict[s]

        episode_reward = 0
        done = False
        c = max_episode_steps
        while c > 0:
            # Render the environment
            # env.render()

            # Agent takes an action using a greedy policy (without exploration)
            action = np.argmax(Q[state])
            next_state, reward, done, _, info = env.step(action.item(),s)
            state = state_dict[next_state]


            episode_reward += reward
            
            if done or c == 1:
                total_rewards.append(episode_reward)
                # print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            c -= 1
    avg_reward = np.mean(total_rewards)
    # print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward


def q_learning(env : GridEnvDeform, num_episodes=1000, max_episode_steps=100, alpha=0.1, gamma=0.99, epsilon=0.1, states_dict=None, evaluate_every=100):
    """
    Perform Q-learning on an environment.
    
    Parameters:
        env: The environment that follows the OpenAI Gym interface.
        num_episodes: Number of episodes to train for.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Exploration rate.
    
    Returns:
        Q-table after training.
    """
    # Initialize Q-table with zeros
    Q = np.zeros((len(env.states), len(env.actions)))
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = states_dict[state]
        done = False
        step = 0

        while not done and step < max_episode_steps:
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                action = np.random.randint(4)  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit
            
            # Take action and observe outcome
            next_state, reward, done, _, _ = env.step(action, execute=True)
            next_state = states_dict[next_state]
            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            Q[state][action] += alpha * (td_target - Q[state][action])
            
            # Update state
            state = next_state
            step += 1
        if episode % evaluate_every == 0:
            avg_reward = eval_tabular(env, Q, states_dict)
            print(f"Episode {episode + 1}/{num_episodes}, Average Reward: {avg_reward}")
            
    epsilon *= 0.99
    if epsilon < 0.1:
        epsilon = 0.1  # Decay epsilon
    
    return Q


"""

alpha learning rate dipendente dallo stato azione a(s,a) = a0/(1+N(s,a)^p) dove N(s,a) è il numero di volte che si è passati per lo stato s e si è scelto l'azione a

analogamente per epsilon


metriche olfactory search panizon celani  



provare inizializzazione con euristiche

"""
