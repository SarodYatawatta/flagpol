#! /usr/bin/env python

import gymnasium as gym
import torch
import numpy as np
import pickle
from flagenv import FlagEnv
from flag_sac import Agent

import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser(
      description='Flagging kernel tuning',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--seed',default=0,type=int,metavar='s',
       help='random seed to use')
    parser.add_argument('--episodes',default=100000,type=int,metavar='g',
       help='number of episodes')
    parser.add_argument('--steps',default=100,type=int,metavar='t',
       help='number of steps per episode')
    args=parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data window (Time,Freq)
    T=5
    F=4

    # actions (determined from the kernels):7+7
    n_actions=14
    # state (kernel error:7+7+7, current action:n_actions:14, data stats:I^2,I^4,Q^2,U^2,V^2(real,imag)=10)
    n_inputs=45
    env=FlagEnv(n_actions=n_actions,T=T,F=F,n_inputs=n_inputs)

    agent=Agent(gamma=0.99, batch_size=256, n_actions=n_actions, alpha=0.03,
       max_mem_size=100000, n_inputs=n_inputs, n_hidden=256, lr_a=0.001, lr_c=0.001)

    scores=[]
    n_games= args.episodes

    # load from previous sessions
    #agent.load_models()
    #with open('scores.pkl','rb') as f:
    #   scores=pickle.load(f)

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        loop=0
        while (not done) and loop<args.steps:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward,
               observation_, done)

            score +=reward
            agent.learn()
            observation = observation_
            loop+=1

        score=score/loop
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)

        if i%500==0:
          # save models to disk
          agent.save_models()

