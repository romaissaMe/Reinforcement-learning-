import numpy as np

#It takes the state-value function of the policy you want to improve, V, and the MDP, P (and gamma, optionally )
#calculate the Q function
#return policy whith actions with highest Q value

def policy_improvement(V,MDP,gamma=1):
    #initialize the Q-function to zero
    Q = np.zeros(len(MDP),len(MDP[0]), dtype=np.float64)
    for s in range(len(MDP)):
        for a in range(len(MDP[s])):
            for prob, next_state, reward, done in MDP[s][a]:
                Q[s][a] += prob * reward + gamma * V[next_state] * (not done)
    
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi


