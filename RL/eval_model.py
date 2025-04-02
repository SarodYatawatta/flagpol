#! /usr/bin/env python

import os
import gymnasium as gym
import torch
import numpy as np
import pickle
from flagenv import FlagEnv
from flag_sac import Agent

import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser(
      description='Flagging kernel evaluation',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--seed',default=0,type=int,metavar='s',
       help='random seed to use')
    parser.add_argument('--episodes',default=100000,type=int,metavar='g',
       help='number of episodes')
    parser.add_argument('--steps',default=10,type=int,metavar='t',
       help='number of steps per episode')
    parser.add_argument('--models',default=4,type=int,metavar='e',
       help='number of models in the ensemble')
    parser.add_argument('--path',default='.', metavar='p',
       help='path (directory) where the models are stored')


    args=parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ensemble models
    N_ens=args.models

    # Data window (Time,Freq)
    T=5
    F=4

    # actions (determined from the kernels)
    n_actions=14
    # state (kernel error, current action, data stats)
    n_inputs=45
    env=FlagEnv(n_actions=n_actions,T=T,F=F,n_inputs=n_inputs)

    # ensemble models
    agents=list()
    for ci in range(N_ens):
      agent=Agent(gamma=0.99, batch_size=256, n_actions=n_actions,
        max_mem_size=100000, n_inputs=n_inputs, n_hidden=256, lr_a=0.001, lr_c=0.001, name_prefix=args.path+os.sep+'run'+str(ci+1)+os.sep)
      agents.append(agent)

    scores=[]
    n_games= args.episodes

    # load from previous sessions
    for agent in agents:
      agent.load_models_for_eval()
    #with open('scores.pkl','rb') as f:
    #   scores=pickle.load(f)

    mean_action=np.zeros(n_actions)
    mean_count=0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        loop=0
        while (not done) and loop<args.steps:
            action_sum=np.zeros(n_actions)
            for agent in agents:
              action = agent.choose_action(observation)
              action_sum +=action
            action_sum = action_sum/N_ens
            action_bit=action_sum/2+0.5
            observation_, reward, done, info = env.step(action_sum)

            score +=reward
            observation = observation_
            loop+=1
        mean_action+=action_bit
        mean_count+=1

        score=score/loop
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)
        print(mean_action/mean_count)
