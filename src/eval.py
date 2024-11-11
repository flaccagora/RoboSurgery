from agents.dqn import DoubleDQNAgent
import numpy as np
from environment.env import GridEnvDeform
import torch
from utils.belief import b_theta_update
from tqdm import trange
import matplotlib.pyplot as plt

def eval_dqn_agent_mdp(agent,env: GridEnvDeform,num_episodes,max_episode_steps,render):
    pass

def eval_dqn_agent_pomdp(agent,env: GridEnvDeform,num_episodes,max_episode_steps,render):
    total_rewards = []

    for episode in range(num_episodes):
        s, _ = env.reset()
        state = torch.tensor([item for sublist in s for item in sublist], dtype=torch.float32)
        
        obs = env.get_observation()

        b_0 = torch.ones(len(env.deformations)) / len(env.deformations)   
        b = b_theta_update(b_0,s[0], obs)
    
        episode_reward = 0
        done = False
        c = 25
        while not done and c > 0:
            # Render the environment
            if render:
                env.render()                                
            
            
            pos = s[0]
            theta = env.deformations[torch.argmax(b)]
            argmaxstate = (pos,theta)
            maxstate = torch.tensor([item for sublist in argmaxstate for item in sublist], dtype=torch.float32)

            # Agent takes an action using a greedy policy (without exploration)
            action = agent.choose_deterministic_action(maxstate)
            next_state, reward, done, _, info = env.step(action,s,execute=render)

            next_obs = env.get_observation(next_state)
            s = next_state

            b_prime = b_theta_update(b,s[0], next_obs)
            b = b_prime

            if render:
                print("State: ", s)
                print("Chosen action: ", action)
                print("Next state: ", next_state)
                print("argmaxstate", argmaxstate)
                print("argmax and max Belief: ", env.deformations[torch.argmax(b_prime)], torch.max(b_prime))
                
            episode_reward += reward
            
            if done or c == 1:
                total_rewards.append(episode_reward)
                # print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            c -= 1
    avg_reward = np.mean(total_rewards)
    # print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    return avg_reward

def eval_agent_pomdp(agent,env: GridEnvDeform,num_episodes,max_episode_steps,render):
    """Returns
        - episode_transition: list of list of tuples (s,a,r,s',done), t[i] is the ith episode
        - beliefs: list of beliefs at each time step 
    """
    if render:
        env.set_rendering()

    transitions = []
    beliefs = []

    for i in trange(num_episodes):

        s, _ = env.reset()
        b = torch.ones(len(env.deformations)) / len(env.deformations)

        totalReward = 0.0
        done = False
        steps = 0
        episode_transitions = []
        episode_beliefs = [b]

        while not done and steps < max_episode_steps:

            best_action = int(agent.get_action(b,s[0]))

            next_state, reward, done, _, info = env.step(best_action, s, execute=render)
            episode_transitions.append((s, best_action, reward, next_state, done))
            next_obs = env.get_observation(next_state)


            totalReward += reward
            next_belief = agent.update_belief(b,next_state[0], next_obs)
            episode_beliefs.append(next_belief)
            

            if render:
                print("State", s)
                print("Action: ", best_action)
                print("Reward:     " + str(totalReward) + "  ")
                print("Next State: ", next_state)
                print("argmax and max Belief: ", env.deformations[torch.argmax(next_belief)], torch.max(next_belief))
                print("Belief entropy: ", agent.get_entropy(next_belief))
                print("\n")
            
                env.render_bis()

            s = next_state
            b = next_belief
            steps += 1

        transitions.append(episode_transitions)
        beliefs.append(episode_beliefs)
    
    if render:
        env.close_render()

    return transitions, beliefs

def eval_agent_mdp(agent,env: GridEnvDeform,num_episodes,max_episode_steps,render):
    """Returns
        - episode_transition: list of list of tuples (s,a,r,s',done), t[i] is the ith episode
        - beliefs: list of beliefs at each time step 
    """
    if render:
        env.set_rendering()

    transitions = []

    for i in trange(num_episodes):

        s, _ = env.reset()

        totalReward = 0.0
        done = False
        steps = 0
        episode_transitions = []

        while not done and steps < max_episode_steps:

            best_action = agent.get_action(s)

            next_state, reward, done, _, info = env.step(best_action, s, execute=render)
            episode_transitions.append((s, best_action, reward, next_state, done))


            totalReward += reward
            

            if render:
                print("State", s)
                print("Action: ", best_action)
                print("Reward:     " + str(totalReward) + "  ")
                print("Next State: ", next_state)
                print("\n")
            
                env.render_bis()

            s = next_state
            steps += 1

        transitions.append(episode_transitions)
    
    if render:
        env.close_render()

    return transitions


def eval_tabular_agent_mdp(agent,env: GridEnvDeform,num_episodes,max_episode_steps,render):
    
    state_dict = env.state_dict

    total_rewards = []

    for episode in range(num_episodes):
        s, _ = env.reset()
        state = state_dict[s]

        episode_reward = 0
        done = False
        c = max_episode_steps
        while not done and c > 0:
            if render:
                env.render()

            # Agent takes an action using a greedy policy (without exploration)
            action = np.argmax(agent[state])
            if render:
                print(f"State: {s}, Action: {action}")
            next_state, reward, done, _, info = env.step(action.item(), s, execute=render)
            state = state_dict[next_state]
            s = next_state

            episode_reward += reward
            
            if done or c == 1:
                total_rewards.append(episode_reward)
                # print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

            c -= 1

    avg_reward = np.mean(total_rewards)

    return total_rewards, avg_reward


def eval_agent(observability,agent,env,num_episodes=100,max_episode_steps=10,render=False):
    if isinstance(agent,DoubleDQNAgent):
        if observability == "MDP":
            return eval_dqn_agent_mdp(agent,env,num_episodes,max_episode_steps,render)
        elif observability == "POMDP":
            return eval_dqn_agent_pomdp(agent,env,num_episodes,max_episode_steps,render)
    else: # assuming infotaxis agent, tabularqwrapper
        if observability == "MDP":
            return eval_agent_mdp(agent,env,num_episodes,max_episode_steps,render)
        elif observability == "POMDP":
            return eval_agent_pomdp(agent,env,num_episodes,max_episode_steps,render)

    return "Invalid agent type"
    

def all_data(transitions, beliefs, path=None):
    # stats on transitions
    completed_episodes = 0
    for i in range(len(transitions)):
        if transitions[i][-1][-1] == 1:
            completed_episodes += 1

    print(f"Completed episodes: {completed_episodes}, out of {len(transitions)}")

    def get_entropy(probabilities):
        
        # Calculate entropy, avoiding log(0) by adding a mask
        entropy = -torch.sum(probabilities * torch.log2(probabilities + (probabilities == 0).float()))
        
        return entropy.item()  # .item() to get a standard Python float

    # plot entropy of beliefs 
    belief_entropy = [[get_entropy(belief) for belief in beliefs_episode] for beliefs_episode in beliefs]
    [plt.plot(belief_entropy[i]) for i in range(len(belief_entropy))]

    # multiple line plot
    plt.title("Entropy of beliefs")
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.legend([f"Episode {i}" for i in range(len(belief_entropy))])

    if path:
        plt.savefig(path + f"entropy_{completed_episodes}out{len(transitions)}.png")
    
    plt.show()