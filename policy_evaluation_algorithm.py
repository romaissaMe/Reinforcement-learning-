import numpy as np

def policy_evaluation(policy, mdp, gamma=1, theta=1e-10):
    state_val = np.zeros(len(mdp)) #we initialize the first itaration estimation of state-value function to zero for all states
    while True:
        V = np.zeros(len(mdp))
        for s in range (len(mdp)): #for each state in env
            for prob, next_state, reward, done in mdp[s][policy[s]] : #take into account each possible transtion from state s in the MDP of env
                V[s] += prob * (reward + gamma * state_val[next_state] * (not done))
        if(np.max(np.abs(state_val - V)) > theta): break
        state_val = V.copy()
    return(V)
        
