import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

# (try to) use a GPU for computation?
use_cuda=True
if use_cuda and torch.cuda.is_available():
  mydevice=torch.device('cuda')
else:
  mydevice=torch.device('cpu')

# action direction: range [0,1], select source if >0.5
LOW=0.0
HIGH=1.

VERBOSE=False
# operation cost, 1 for 32-bit
COST32=1
COST16=0.6
COSTCAST=0.3 # casting from 32 to 16 bit and vice versa

# max state value (will take log() of this)
MAXVAL=100000
# penalty for nan/inf
PENALTY=20
# reward for getting flags right or wrong (-reward)
REWARD=100

class FlagEnv(gym.Env):
    metadata={'render.modes':['human']}


    # n_actions=action space
    # time/freq window TxF
    # each point on window [2x2] complex data (4 correlations)
    def __init__(self,n_actions=14,n_inputs=45,T=5,F=4):
        super(FlagEnv,self).__init__()
        self.n_actions=n_actions
        self.T=T
        self.F=F
        # maximum scale of the data
        self.max_scale=1000
        # when to generate RFI, if U(0,1) > value given below
        self._rfi_prob=0.4
        self.EPS=1e-6

        # actions, divided into SK,DS
        self._K_SK=7
        self._K_DS=7
        assert(self.n_actions==self._K_SK+self._K_DS)

        # actions [0,1]^n_actions -> scaled to [-1,1]^n_actions
        self.action_space=spaces.Box(low=np.ones((self.n_actions,1))*(-1),high=np.ones((self.n_actions,1))*(1),dtype=np.float32)

        # observation: 
        # - normalized TF data, log(abs(max)) of data (normalization factor), = T*F*8+1
        # - error in each statement (finite precision statement - full precision statement) = ??
        # - [0,1]^n_actions (current action) = n_actions
        self.n_inputs=n_inputs
        self.observation_space=spaces.Box(low=np.ones((self.n_inputs,1))*(-1000),high=np.ones((self.n_inputs,1))*(1000),dtype=np.float32)

    def _generate_data(self):
        scalefactor=np.random.rand(1)*self.max_scale
        self.Ir=np.random.rand(self.T,self.F)*scalefactor
        self.Ii=np.random.rand(self.T,self.F)*scalefactor
        self.Qr=np.random.rand(self.T,self.F)*scalefactor
        self.Qi=np.random.rand(self.T,self.F)*scalefactor
        self.Ur=np.random.rand(self.T,self.F)*scalefactor
        self.Ui=np.random.rand(self.T,self.F)*scalefactor
        self.Vr=np.random.rand(self.T,self.F)*scalefactor
        self.Vi=np.random.rand(self.T,self.F)*scalefactor

        # add RFI
        if np.random.rand()>self._rfi_prob:
            self._add_rfi_to_data()

        # data (sufficient) statistics log () mean I^2,I^4,Q^2,U^2,V^2 (real,imag)
        self.data_stats=np.zeros(10)
        self.data_stats[0]=np.log(np.sum(self.Ir.flatten()**2)/(self.T*self.F))
        self.data_stats[1]=np.log(np.sum(self.Ir.flatten()**4)/(self.T*self.F))
        self.data_stats[2]=np.log(np.sum(self.Qr.flatten()**2)/(self.T*self.F))
        self.data_stats[3]=np.log(np.sum(self.Ur.flatten()**2)/(self.T*self.F))
        self.data_stats[4]=np.log(np.sum(self.Vr.flatten()**2)/(self.T*self.F))
        self.data_stats[5]=np.log(np.sum(self.Ii.flatten()**2)/(self.T*self.F))
        self.data_stats[6]=np.log(np.sum(self.Ii.flatten()**4)/(self.T*self.F))
        self.data_stats[7]=np.log(np.sum(self.Qi.flatten()**2)/(self.T*self.F))
        self.data_stats[8]=np.log(np.sum(self.Ui.flatten()**2)/(self.T*self.F))
        self.data_stats[9]=np.log(np.sum(self.Vi.flatten()**2)/(self.T*self.F))


    def _add_rfi_to_data(self):
        # generate 2x2 matrices (real,imag)
        scalefactor=np.random.rand(1)*self.max_scale
        J=np.random.rand(4)*scalefactor
        t_low=np.random.randint(0,self.T)
        t_high=np.random.randint(t_low,self.T)
        f_low=np.random.randint(0,self.F)
        f_high=np.random.randint(f_low,self.F)
        t_w=t_high-t_low+1
        f_w=f_high-f_low+1
        self.Ir[t_low:t_high+1,f_low:f_high+1]+=np.ones((t_w,f_w))*scalefactor*J[0]
        self.Qr[t_low:t_high+1,f_low:f_high+1]+=np.ones((t_w,f_w))*scalefactor*J[1]
        self.Ur[t_low:t_high+1,f_low:f_high+1]+=np.ones((t_w,f_w))*scalefactor*J[2]
        self.Vr[t_low:t_high+1,f_low:f_high+1]+=np.ones((t_w,f_w))*scalefactor*J[3]

        scalefactor=np.random.rand(1)*self.max_scale
        J=np.random.rand(4)*scalefactor
        t_low=np.random.randint(0,self.T)
        t_high=np.random.randint(t_low,self.T)
        f_low=np.random.randint(0,self.F)
        f_high=np.random.randint(f_low,self.F)
        t_w=t_high-t_low+1
        f_w=f_high-f_low+1
        self.Ii[t_low:t_high+1,f_low:f_high+1]+=np.ones((t_w,f_w))*scalefactor*J[0]
        self.Qi[t_low:t_high+1,f_low:f_high+1]+=np.ones((t_w,f_w))*scalefactor*J[1]
        self.Ui[t_low:t_high+1,f_low:f_high+1]+=np.ones((t_w,f_w))*scalefactor*J[2]
        self.Vi[t_low:t_high+1,f_low:f_high+1]+=np.ones((t_w,f_w))*scalefactor*J[3]



    def _random_action(self):
        action=np.rint(np.random.rand(self.n_actions))
        action_SK=action[0:self._K_SK]
        action_DS=action[self._K_SK:self.n_actions]
        return action_SK,action_DS

    def kernel_spectral_kurtosis(self,Xr,Xi,action):
        # X: complex, T x 1 vector, Xr,Xi: real/imaginary parts
        # action: Kx1, K=7
        # return : state (Kx1), cost 1x1, reward 1x1
        M=Xr.shape[0] # averaging length
    
        # cumulative cost
        cost=0
    
        # reward
        reward=0
    
        # state: Kx1
        state=torch.zeros(action.shape)
    
        # find (|X|^2)
        TYPE=torch.float32 if action[0]==0 else torch.float16
        X2=Xr.to(TYPE)**2+Xi.to(TYPE)**2 # st 0
        if TYPE==torch.float32:
          cost +=2*M*COST32
        elif TYPE==torch.float16:
          cost +=2*M*COST16
        X2_=Xr.to(torch.float64)**2+Xi.to(torch.float64)**2
        state[0]=(X2.to(torch.float64)-X2_).abs().sum()
        if VERBOSE:
          print(TYPE)
          print(state[0])

        # find abs(X)=sqrt(|X|^2)
        TYPE=torch.float32 if action[1]==0 else torch.float16
        Xa=torch.sqrt(X2).to(TYPE) # st 1
        if TYPE==torch.float32:
          cost +=M*COST32
        elif TYPE==torch.float16:
          cost +=M*COST16
        Xa_=torch.sqrt(X2_.to(torch.float64))
        state[1]=(Xa.to(torch.float64)-Xa_).abs().sum()
        # incremental cost if type changed
        if (action[0] != action[1]) :
           cost +=M*COSTCAST
        if VERBOSE:
          print(TYPE)
          print(state[1])

        # find S1=sum(abs(X))
        TYPE=torch.float32 if action[2]==0 else torch.float16
        S1=torch.sum(Xa).to(TYPE) # st 2
        if TYPE==torch.float32:
          cost +=M*COST32
        elif TYPE==torch.float16:
          cost +=M*COST16
        S1_=torch.sum(Xa_).to(torch.float64)
        state[2]=(S1.to(torch.float64)-S1_).abs().sum()
        if (action[1] != action[2]) :
           cost +=M*COSTCAST
        if VERBOSE:
          print(TYPE)
          print(state[2])

        # X4=abs(X)^2 = X2

        # S2=sum(X4) = sum(X2)
        TYPE=torch.float32 if action[3]==0 else torch.float16
        S2=torch.sum(X2).to(TYPE) # st 4
        if TYPE==torch.float32:
          cost +=M*COST32
        elif TYPE==torch.float16:
          cost +=M*COST16
        S2_=torch.sum(X2_).to(torch.float64)
        state[3]=(S2.to(torch.float64)-S2_).abs().sum()
        if (action[1] != action[3]) :
           cost +=M*COSTCAST
        if VERBOSE:
          print(TYPE)
          print(state[3])

        TYPE=torch.float32 if action[4]==0 else torch.float16
        SK_d=0.5  # st 4 - constants
        EPS=1e-6 # st 4
        SK_low=torch.tensor(0.001).to(TYPE) # st 4
        SK_high=torch.tensor(3.3).to(TYPE) # st 4
        SK_low_=torch.tensor(0.001).to(TYPE)
        SK_high_=torch.tensor(3.3).to(TYPE)
        state[4]=(SK_high-SK_high_).abs()+(SK_low-SK_low_).abs()
    
        TYPE=torch.float32 if action[5]==0 else torch.float16
        S12=(S1*S1).to(TYPE)+torch.tensor(EPS).to(TYPE) # st 5
        if TYPE==torch.float32:
            cost +=COST32
        elif TYPE==torch.float16:
            cost +=COST16
        S12_=(S1_*S1_+EPS).to(torch.float64)
        state[5]=(S12.to(torch.float64)-S12_).abs().sum()
        if VERBOSE:
          print(TYPE)
          print(state[5])
    
        TYPE=torch.float32 if action[6]==0 else torch.float16
        t=((M*SK_d+1)/(M-1))*(M*S2/(S12)-1).to(TYPE) # st 6
        if TYPE==torch.float32:
            cost +=COST32
        elif TYPE==torch.float16:
            cost +=COST16
        t_=((M*SK_d+1)/(M-1))*(M*S2_/(S12_)-1).to(torch.float64)
        state[6]=(t.to(torch.float64)-t_).abs().sum()
        if VERBOSE:
          print(TYPE)
          print(state[6])

        fll=(t<SK_low) # st 4
        fll_=(t_<SK_low_)
        fhh=(t>SK_high) # st 4
        fhh_=(t_>SK_high_)
        if TYPE==torch.float32:
            cost +=(1+1)*COST32
        elif TYPE==torch.float16:
            cost +=(1+1)*COST16
        flags=fll | fhh
        flags_=fll_ | fhh_
        if VERBOSE:
          print(flags)
          print(flags_)

        if (flags==flags_):
            reward+=REWARD
        else:
            reward=-REWARD
        
        state[state==0]=self.EPS
        # if any state is not finite, add penalty
        penalty=0
        if (not torch.all(torch.isfinite(state))):
            penalty -=PENALTY

        state=np.log(np.nan_to_num(state.numpy(),nan=MAXVAL,posinf=MAXVAL,neginf=MAXVAL))/np.log(MAXVAL)
        return state,cost,reward+penalty

    def kernel_dir_statistics(self,Qin,Uin,Vin,action):
        # Qin,Uin,Vin: real, T x 1 vectors
        # action: Kx1, K=7
        # return : state (Kx1), cost 1x1, reward 1x1
        N=Qin.shape[0] # averaging length

        # cumulative cost
        cost=0

        # reward
        reward=0

        # state: Kx1
        state=torch.zeros(action.shape)

        TYPE=torch.float32 if action[0]==0 else torch.float16
        Q2=Qin.to(TYPE)**2 # st 0
        U2=Uin.to(TYPE)**2 # st 0
        V2=Vin.to(TYPE)**2 # st 0
        if TYPE==torch.float32:
            cost +=N*COST32
        elif TYPE==torch.float16:
            cost +=N*COST16
        Q2_=Qin.to(torch.float64)**2
        U2_=Uin.to(torch.float64)**2
        V2_=Vin.to(torch.float64)**2
        state[0]=(Q2.to(torch.float64)-Q2_).abs().sum()\
          +(U2.to(torch.float64)-U2_).abs().sum()\
          +(V2.to(torch.float64)-V2_).abs().sum()
        if VERBOSE:
          print(TYPE)
          print(state[0])


        TYPE=torch.float32 if action[1]==0 else torch.float16
        pp=torch.sqrt((Q2+U2+V2).to(TYPE)).to(TYPE) # st 1
        if TYPE==torch.float32:
            cost +=2*N*COST32
        elif TYPE==torch.float16:
            cost +=2*N*COST16
        pp_=torch.sqrt((Q2_+U2_+V2_).to(torch.float64)).to(torch.float64)
        state[1]=(pp.to(torch.float64)-pp_).abs().sum()
        if (action[0] != action[1]) :
          cost +=N*COSTCAST
        if VERBOSE:
          print(TYPE)
          print(state[1])

        TYPE=torch.float32 if action[2]==0 else torch.float16
        Q=(Qin.to(TYPE)/pp.to(TYPE)).to(TYPE)# st 2
        U=(Uin.to(TYPE)/pp.to(TYPE)).to(TYPE)# st 2
        V=(Vin.to(TYPE)/pp.to(TYPE)).to(TYPE)# st 2
        if TYPE==torch.float32:
            cost +=(N)*COST32
        elif TYPE==torch.float16:
            cost +=(N)*COST16
        Q_=(Qin/pp_).to(torch.float64)
        U_=(Uin/pp_).to(torch.float64)
        V_=(Vin/pp_).to(torch.float64)
        state[2] +=(Q.to(torch.float64)-Q_).abs().sum()\
          +(U.to(torch.float64)-U_).abs().sum()\
          +(V.to(torch.float64)-V_).abs().sum()
        if (action[1] != action[2]) :
          cost +=N*COSTCAST
        if VERBOSE:
          print(TYPE)
          print(state[2])

        TYPE=torch.float32 if action[3]==0 else torch.float16
        # sum whole block into scalar
        qq=torch.sum(Q).to(TYPE) # st 3
        uu=torch.sum(U).to(TYPE) # st 3
        vv=torch.sum(V).to(TYPE) # st 3
        if TYPE==torch.float32:
            cost +=(N)*COST32
        elif TYPE==torch.float16:
            cost +=(N)*COST16
        qq_=torch.sum(Q_).to(torch.float64)
        uu_=torch.sum(U_).to(torch.float64)
        vv_=torch.sum(V_).to(torch.float64)
        state[3]=(qq.to(torch.float64)-qq_).abs().sum()\
          +(uu.to(torch.float64)-uu_).abs().sum()\
          +(vv.to(torch.float64)-vv_).abs().sum()
        if (action[2] != action[3]) :
          cost +=N*COSTCAST
        if VERBOSE:
          print(TYPE)
          print(state[3])

        TYPE=torch.float32 if action[4]==0 else torch.float16
        qq2=qq.to(TYPE)**2 # st 4
        uu2=uu.to(TYPE)**2 # st 4
        vv2=vv.to(TYPE)**2 # st 4
        if TYPE==torch.float32:
            cost +=COST32
        elif TYPE==torch.float16:
            cost +=COST16
        qq2_=qq_.to(torch.float64)**2
        uu2_=uu_.to(torch.float64)**2
        vv2_=vv_.to(torch.float64)**2
        state[4]=(qq2.to(torch.float64)-qq2_).abs().sum()\
          +(uu2.to(torch.float64)-uu2_).abs().sum()\
          +(vv2.to(torch.float64)-vv2_).abs().sum()
        if (action[3] != action[4]) :
          cost +=COSTCAST
        if VERBOSE:
          print(TYPE)
          print(state[4])

        TYPE=torch.float32 if action[5]==0 else torch.float16
        rr=(qq2+uu2+vv2).to(TYPE) # st 5
        rvec=torch.sqrt(rr)/N # st 5
        if TYPE==torch.float32:
            cost +=(1+1)*COST32
        elif TYPE==torch.float16:
            cost +=(1+1)*COST16
        rr_=(qq2_+uu2_+vv2_).to(torch.float64)
        rvec_=torch.sqrt(rr_)/N
        state[5]=(rvec.to(torch.float64)-rvec_).abs().sum()
        if (action[4] != action[5]) :
          cost +=COSTCAST
        if VERBOSE:
          print(TYPE)
          print(state[5])

        TYPE=torch.float32 if action[6]==0 else torch.float16
        DS_gamma=torch.tensor(0.95).to(TYPE) # st 6 - constants
        flags=rvec > DS_gamma # st 6
        if TYPE==torch.float32:
            cost +=(1+1)*COST32
        elif TYPE==torch.float16:
            cost +=(1+1)*COST16
        flags_=rvec_ > DS_gamma.to(torch.float64)
        state[6]=DS_gamma-DS_gamma.to(torch.float64)
        if VERBOSE:
          print(TYPE)
          print(flags)
          print(flags_)

        if (flags==flags_):
            reward+=REWARD
        else:
            reward=-REWARD
 
        state[state==0]=self.EPS
        # if any state is not finite, add penalty
        penalty=0
        if (not torch.all(torch.isfinite(state))):
            penalty -=PENALTY

        state=np.log(np.nan_to_num(state.numpy(),nan=MAXVAL,posinf=MAXVAL,neginf=MAXVAL))/np.log(MAXVAL)
        return state,cost,reward+penalty

    def step(self,action):
        done=False
        # action [-1,1]^n_actions -> map to [0,1]^n_actions
        action_cont=action*(HIGH-LOW)/2+(HIGH+LOW)/2
        # discretize to [0,1]
        action_bins=np.rint(action_cont)
        action_SK=action_bins[0:self._K_SK]
        action_DS=action_bins[self._K_SK:self.n_actions]

        # evaluate action
        state1,cost1,reward1=self.kernel_spectral_kurtosis(torch.tensor(self.Ir.flatten()),torch.tensor(self.Ii.flatten()),action_SK)
        state2,cost2,reward2=self.kernel_dir_statistics(torch.tensor(self.Qr.flatten()),torch.tensor(self.Ur.flatten()),torch.tensor(self.Vr.flatten()),action_DS)
        state3,cost3,reward3=self.kernel_dir_statistics(torch.tensor(self.Qi.flatten()),torch.tensor(self.Ui.flatten()),torch.tensor(self.Vi.flatten()),action_DS)
        observation=np.concatenate((state1,state2,state3,action_SK,action_DS))

        # calculate reward: scale reward up by x3 compared to cost
        reward=(reward1+(reward2+reward3)/2)*0.3 -(cost1+(cost2+cost3)/2)*0.1

        # form observation: data+action_bins+statistics
        observation=np.concatenate((state1,state2,state3,action_SK,action_DS,self.data_stats))

        info={}

        return observation, reward, done, info


    def reset(self):
        # generate new data
        self._generate_data()
        # use default action = random bits
        action_SK,action_DS=self._random_action()

        state1,cost1,reward1=self.kernel_spectral_kurtosis(torch.tensor(self.Ir.flatten()),torch.tensor(self.Ii.flatten()),action_SK)
        state2,cost2,reward2=self.kernel_dir_statistics(torch.tensor(self.Qr.flatten()),torch.tensor(self.Ur.flatten()),torch.tensor(self.Vr.flatten()),action_DS)
        state3,cost3,reward3=self.kernel_dir_statistics(torch.tensor(self.Qi.flatten()),torch.tensor(self.Ui.flatten()),torch.tensor(self.Vi.flatten()),action_DS)
        observation=np.concatenate((state1,state2,state3,action_SK,action_DS,self.data_stats))

        return observation

    def render(self,mode='human'):
        print('Render not implemented')


    def close(self):
        pass


#env=FlagEnv(n_actions=14,T=5,F=4,n_inputs=45)
#env.reset()
#action=(np.random.rand(14)-0.5)*2
#obs,reward,done,_=env.step(action)
#print(reward)
#print(obs.shape)
