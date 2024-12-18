#! /usr/bin/env python

import math,sys,uuid
import argparse
from multiprocessing import Pool
from multiprocessing import shared_memory
import numpy as np
from scipy import special as sp
from scipy.io import savemat
import casacore.tables as ct
from lbfgsb import LBFGSB
import torch # only for finite precision

EPS=1e-6
UP_DS=1.5 # factor to increase DS threshold
UP_SK=1.0 # factor to increase SK threshold

def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

def qfunc(x):
    return 0.5-0.5*sp.erf(x/np.sqrt(2))

def invQfunc(x):
    return np.sqrt(2)*sp.erfinv(1-2*x)

def coth(x):
    return 1.0/(np.tanh(x)+EPS)

def csch(x):
    return 1.0/(np.sinh(x)+EPS)

## spectral kurtosis upper/lower bounds
def gsk_bounds(M,d,PFA):
    # M : offline accumulation
    # N : online accumulation = 1
    # d : d=0.5, (time domain, chi-squared dist), d=1 (spectral domain, exponential dist)
    # PFA : false alarm
    N=1
    x=np.arange(0.001,20,0.1)
    Nd=N*d
    m1=1
    m2=(2*( M**2)* Nd *(1 + Nd))/((-1 + M) *(6 + 5* M* Nd + (M**2)*( Nd**2)))
    m3=(8*( M**3)* Nd* (1 + Nd)* (-2 + Nd* (-5 + M *(4 + Nd))))/(((-1 + M)**2)* (2 + M* Nd) *(3 +M* Nd)* (4 + M* Nd)* (5 + M* Nd))
    mm=(-(m3-2*m2**2)/m3+x)/(m3/2/m2)
    mm[mm<0.0]=EPS
    yup=np.abs((1.0-sp.gammainc(4*(m2**3/(m3**2)),mm))-PFA)
    ylow=np.abs(sp.gammainc(4*(m2**3/(m3**2)),mm)-PFA)
    id1=np.argmin(ylow)
    id2=np.argmin(yup)

    return (x[id1],x[id2])


class PolFlagger:

    def __init__(self,msname,method,colname,preserve,do_debug=False,do_update=False,parallel_jobs=2,freq_window_size=2,time_window_size=10,admm_rho=0.001,admm_iter=10,finite_prec=False):
        self._ms=msname
        self._method=method
        self._col=colname
        self._preserve_flags=preserve
        self._tab=None # table (for reading)
        self._tab_w=None # table for writing
        self._T=time_window_size # time window
        self._F=1 # no. of frequencies (=no. of data points = T x F)
        self._freq=None # frequencies
        self._Nant=1 # antennas
        self._B=1 # baselines (including autocorrelations)
        self._Npar=parallel_jobs # parallel jobs
        self._debug=do_debug # if true, run in sequential mode (not parallel)
        self._W=freq_window_size # window size (in frequency) to eval directional stats
        self._finite_prec=finite_prec # if true, use finite precision arithmetic, defined by the action below
        self._finite_act_sk=[0, 0, 0, 0, 1, 0, 1] # =7
        self._finite_act_ds=[0, 0, 0, 1, 1, 1, 1] # =7

        self.get_metadata()

        # for update (solving)
        self._update=do_update
        self._solver=None

        # how many windows? (upper bound)
        self._D=(self._F+self._W-1)//self._W
        self._N=self._W*self._T # data points used in averaging freq x time window size
        self._p=3 # parameters for finding concentration parameter kappa (k)
        # default false alarm probability
        self._Pf=0.01
        self._g_low=0
        self._g=None
        self._eps=EPS
        self._r_max=0.98 # making close to 1 results in NaNs
        print(f'divide {self._F} channels into {self._D} windows of size {self._T}x{self._W}')

        self.set_default_values()

        # consensus polynomial
        self._cP=3 # number of basis functions
        self._ADMM=admm_iter # ADMM iterations
        self._rho=admm_rho # ADMM rho
        self._cB=None
        self._cBi=None
        # global variable
        self._z=np.zeros((self._cP,1))
        # Lagrange multiplier
        self._y=np.zeros((self._D,1))
        self.create_consensus_poly()

        # parameters for spectral kurtosis (for corss correlations, need to be empirically determined using noise only data)
        self._sk_d=1.0/2 # d=0.5, (time domain, chi-squared dist), d=1 (spectral domain, exponential dist)
        (sklow,skhigh)=gsk_bounds(self._N,self._sk_d,self._Pf)
        self._sk_low=sklow*UP_SK
        self._sk_high=skhigh*UP_SK
        print(f'Default SK {self._Pf} threshold {self._sk_low} {self._sk_high}')
        print(f'Default DS {self._Pf} threshold {self._F} {self._g_low}')


    def open_ms(self):
        self._tab=ct.table(self._ms,readonly=True)
        self._tab_w=ct.table(self._ms,readonly=False)

    def create_consensus_poly(self):
      # setup consensus polynomials
      # select frequencies for evaluating basis
      ffreq=self._freq[0,0:self._F:self._W]
      f0=np.mean(ffreq)
      assert(self._D==ffreq.size)
      B=np.zeros((self._cP,self._D))
      B[0]=np.ones(self._D)
      B[1]=(ffreq-f0)/f0
      B[2]=((ffreq-f0)/f0)**2
      # normalize each basis to have range in [-1,1]
      B[1]=B[1]/np.abs(B[1]).max()
      B[2]=B[2]/np.abs(B[2]).max()
      self._cB=B
      Bi=np.zeros((self._cP,self._cP))
      for ci in range(self._D):
          Bi=Bi+self._rho*np.outer(B[:,ci],B[:,ci])
      self._cBi=np.linalg.pinv(Bi)

    def set_default_values(self):
      P_1=1.856697
      P_2=0.283628;
      # formula is valid for 20 < N < 1000
      self._g_low=np.sqrt(self._N/(3*P_2)*invQfunc(self._Pf/P_1)) # multiplier hand-tuned
      # normalize by 1/N (only whan flagging) because directional statistic is also normalized, also use a (higher=UP_DS) default value than the lower bound
      self._g=np.ones(self._D)*self._g_low*UP_DS

      # if solving, create persistent solver
      if self._update:
         # set lower/upper bounds
         g_low=np.ones((self._D,1))*self._g_low*0.9
         g_high=np.ones((self._D,1))*self._g_low*10.0
         x0=np.zeros((self._D,1))
         x0[:,0]=self._g
         self._solver=LBFGSB(x0,g_low,g_high,max_iter=4,batch_mode=True)

    def close_ms(self):
        self._tab_w.close()
        self._tab.close()

    def write_flags(self):
        pass

    def get_metadata(self):
        # get number of frequencies, antennas etc.
        tf=ct.table(self._ms+'/SPECTRAL_WINDOW',readonly=True)
        ch=tf.getcol('CHAN_FREQ')
        self._F=ch.shape[1]
        self._Nant=ct.table(self._ms+'/ANTENNA').nrows()
        self._B=self._Nant*(self._Nant+1)//2
        self._freq=ch


    def directional_statistics(self,ii,qq,uu,vv):
        # choose full precision or finite precision version
        if self._finite_prec:
            return self.directional_statistics_fp_(ii,qq,uu,vv)
        else :
            return self.directional_statistics_(ii,qq,uu,vv) 

    def directional_statistics_fp_(self,ii,qq,uu,vv):
            # ii,qq,uu,vv: TxFx1 (real) vectors
            # return flags, and extra info if needed

            TYPE=torch.float32 if self._finite_act_ds[0]==0 else torch.float16
            # create torch tensors
            qq=torch.tensor(qq).to(TYPE)
            uu=torch.tensor(uu).to(TYPE)
            vv=torch.tensor(vv).to(TYPE)

            q2=qq**2
            u2=uu**2
            v2=vv**2

            TYPE=torch.float32 if self._finite_act_ds[1]==0 else torch.float16
            pp=torch.sqrt((q2+u2+v2).to(TYPE)).to(TYPE)
            # make sure all pp are not zero
            pp[pp<self._eps]=self._eps

            TYPE=torch.float32 if self._finite_act_ds[2]==0 else torch.float16
            qq=(qq.to(TYPE)/pp).to(TYPE)
            uu=(uu.to(TYPE)/pp).to(TYPE)
            vv=(vv.to(TYPE)/pp).to(TYPE)

            TYPE=torch.float32 if self._finite_act_ds[3]==0 else torch.float16
            # sum blocks of W to map 1:F to 1:D
            qq1=torch.zeros(self._D).to(TYPE)
            uu1=torch.zeros(self._D).to(TYPE)
            vv1=torch.zeros(self._D).to(TYPE)
            for ci in range(self._D):
                qq1[ci]=torch.sum(qq[:,self._W*ci:self._W*(ci+1)]).to(TYPE)
                uu1[ci]=torch.sum(uu[:,self._W*ci:self._W*(ci+1)]).to(TYPE)
                vv1[ci]=torch.sum(vv[:,self._W*ci:self._W*(ci+1)]).to(TYPE)

            TYPE=torch.float32 if self._finite_act_ds[4]==0 else torch.float16
            qq1=(qq1**2).to(TYPE)
            uu1=(uu1**2).to(TYPE)
            vv1=(vv1**2).to(TYPE)

            TYPE=torch.float32 if self._finite_act_ds[5]==0 else torch.float16
            rr=(qq1+uu1+vv1).to(TYPE)
            # find length, normalize with window size
            Rvec=torch.sqrt(rr)/self._N
            # replace NaNs
            Rvec[torch.isnan(Rvec)]=10*self._r_max

            TYPE=torch.float32 if self._finite_act_ds[6]==0 else torch.float16
            flags=(Rvec > torch.tensor(self._g/self._N).to(TYPE)).numpy()

            return flags

    def directional_statistics_(self,ii,qq,uu,vv):
            # ii,qq,uu,vv: TxFx1 (real) vectors
            # return flags, and extra info if needed

            # FIXME: replace pp with online exponential average ???
            pp=np.sqrt(qq**2+uu**2+vv**2)
            # make sure all pp are not zero
            pp[pp<self._eps]=self._eps
            qq=qq/pp
            uu=uu/pp
            vv=vv/pp

            # sum blocks of W to map 1:F to 1:D
            qq1=np.zeros(self._D)
            uu1=np.zeros(self._D)
            vv1=np.zeros(self._D)
            for ci in range(self._D):
                qq1[ci]=np.sum(qq[:,self._W*ci:self._W*(ci+1)])
                uu1[ci]=np.sum(uu[:,self._W*ci:self._W*(ci+1)])
                vv1[ci]=np.sum(vv[:,self._W*ci:self._W*(ci+1)])

            rr=qq1**2+uu1**2+vv1**2
            # find length, normalize with window size
            Rvec=np.sqrt(rr)/self._N
            if (self._update):
              # find kappa (default is a large value)
              kappa=np.ones(self._D)*1000
              for ci in range(self._D):
                 if Rvec[ci]<self._r_max and Rvec[ci]>self._eps:
                    k1=Rvec[ci]*(self._p-Rvec[ci]**2)/(1-Rvec[ci]**2)
                    Ak=sp.iv(self._p/2,k1)/sp.iv((self._p/2-1),k1)
                    k2=k1-(Ak-Rvec[ci])/(1-Ak**2-(self._p-1)/k1*Ak)
                    Ak=sp.iv(self._p/2,k2)/sp.iv((self._p/2-1),k2)
                    k3=k2-(Ak-Rvec[ci])/(1-Ak**2-(self._p-1)/k2*Ak)
                    kappa[ci]=k3
              kappa[np.isnan(kappa)]=1000

            # replace NaNs
            Rvec[np.isnan(Rvec)]=10*self._r_max

            flags=(Rvec > self._g/self._N)
            if self._update:
              return flags,kappa
            else:
              return flags

    def apply_flags(self,data):
        # in: data: TxFx4, data for one baseline
        # average data over T
        # flagging method: determined by self
        # return: flags 1xFx4 (bool), 
        # and extra data depending on fitting is set or not 

        # resultant vector for (real, imag) data
        Rvec=np.zeros((self._D,1))
        Rveci=np.zeros((self._D,1))
        # concentration parameter
        Kvec=np.zeros((self._D,1))
        Kveci=np.zeros((self._D,1))

        # map to Poincare sphere all F channels
        XX=data[:,:,0]
        XY=data[:,:,1]
        YX=data[:,:,2]
        YY=data[:,:,3]
        sI=XX+YY
        sQ=XX-YY
        sU=XY+YX
        sV=1j*(XY-YX)


        # threshold rr,rr_i to find flags, also flag NaN
        if self._method=='DS' or self._method=='BOTH':
            # now process real, imag parts separately
            stokes_i=np.real(sI);
            stokes_q=np.real(sQ);
            stokes_u=np.real(sU);
            stokes_v=np.real(sV);
            if self._update:
              flag_real,kappa=self.directional_statistics(stokes_i,stokes_q,stokes_u,stokes_v)
            else:
              flag_real=self.directional_statistics(stokes_i,stokes_q,stokes_u,stokes_v)
            stokes_i=np.imag(sI);
            stokes_q=np.imag(sQ);
            stokes_u=np.imag(sU);
            stokes_v=np.imag(sV);
            if self._update:
              flag_imag,kappa_i=self.directional_statistics(stokes_i,stokes_q,stokes_u,stokes_v)
            else:
              flag_imag=self.directional_statistics(stokes_i,stokes_q,stokes_u,stokes_v)

            flag_both = (flag_real | flag_imag) # use & or | here

        flag=np.zeros((self._F,4),dtype=bool)

        if self._method=='SK4':
              flag_I=self.flag_spectral_kurtosis(sI)
              flag_Q=self.flag_spectral_kurtosis(sQ)
              flag_U=self.flag_spectral_kurtosis(sU)
              flag_V=self.flag_spectral_kurtosis(sV)
              flag=(flag | flag_I)
              flag=(flag | flag_Q)
              flag=(flag | flag_U)
              flag=(flag | flag_V)

        if (self._method=='DS' or self._method=='BOTH') :
              # expand flags from 1:D to 1:F
              for ci in range(self._D):
                  if flag_both[ci]:
                      flag[self._W*ci:self._W*(ci+1),:]=True
        if (self._method=='SK' or self._method=='BOTH') :
              # use Stokes I spectral kurtosis to find flags
              flag_sk=self.flag_spectral_kurtosis(sI)
              flag=(flag | flag_sk)

        if not (self._debug or self._update):
           return flag
        elif self._debug and not self._update:
           return flag,rr,rr_i
        elif self._update:
           return flag,rr,rr_i,kappa,kappa_i

    def iter_ms(self):
        self.open_ms()

        if self._debug:
            counter=0

        # outer iteration over time
        do_iterate=True
        iter_r=self._tab.iter(['TIME'])
        iter_w=self._tab_w.iter(['TIME'])
        while do_iterate:
          try:
          #########################################
            n_pol=4
            data_t=np.zeros((self._T,self._B,self._F,n_pol),dtype=complex)
            flag_t=np.zeros((self._T,self._B,self._F,n_pol),dtype=bool)
            for nt in range(self._T):
               t1=iter_r.next()
               tt=t1.query(sortlist='ANTENNA1,ANTENNA2',columns='ANTENNA1,ANTENNA2,FLAG,'+self._col)
               # shape: n_baselines x n_freq x n_pol(=4)
               data=tt.getcol(self._col)
               data_t[nt]=data
               flag=tt.getcol('FLAG')
               a1=tt.getcol('ANTENNA1')
               a2=tt.getcol('ANTENNA2')
               autocorr=(a1==a2)
            # create shared mem
            shm_flag=shared_memory.SharedMemory(create=True,size=flag.nbytes)
            flag_sh=np.ndarray(flag.shape,dtype=flag.dtype,buffer=shm_flag.buf)

            # for debugging/solving
            rvec=np.zeros((self._B,self._D,2))
            if self._debug or self._update:
                   shm_rvec=shared_memory.SharedMemory(create=True,size=rvec.nbytes)
                   rvec_sh=np.ndarray(rvec.shape,dtype=rvec.dtype,buffer=shm_rvec.buf)
            # for solving
            kappa=np.zeros((self._B,self._D,2))
            if self._update:
                   shm_kappa=shared_memory.SharedMemory(create=True,size=kappa.nbytes)
                   kappa_sh=np.ndarray(kappa.shape,dtype=kappa.dtype,buffer=shm_kappa.buf)
            # local function for parallel jobs
            @globalize
            def flag_baseline(index):
                   vs=data_t[:,index,:,:]
                   if not (self._debug or self._update):
                       flags=self.apply_flags(vs)
                   elif self._debug and not self._update:
                       flags, gamma, gamma_i=self.apply_flags(vs)
                   elif self._update:
                       flags, gamma, gamma_i, kappa, kappa_i=self.apply_flags(vs)

                   flag_sh[index,:,:]=flags
                   if self._debug or self._update:
                       rvec_sh[index,:,0]=gamma
                       rvec_sh[index,:,1]=gamma_i
                   if self._update:
                       kappa_sh[index,:,0]=kappa
                       kappa_sh[index,:,1]=kappa_i


            # iter over baselines (done in parallel)
            if not self._debug:
                 pool=Pool(self._Npar)
                 pool.map(flag_baseline,range(self._B))
                 pool.close()
                 pool.join()
            else: # debug mode, for debugging/plotting
                 for nb in range(self._B):
                     flag_baseline(nb)
                   
            # preserve old flags
            if self._preserve_flags:
                   flag[:] |= flag_sh[:]
            else:
                   flag[:] = flag_sh[:]
            shm_flag.close()
            shm_flag.unlink()

            for nt in range(self._T):
               flag_t[nt]=flag

            if self._debug and not self._update:
                   rvec[:]=rvec_sh[:]
                   shm_rvec.close()
                   shm_rvec.unlink()
                   mydict={'rvec':rvec}
                   savemat('gg_'+str(counter)+'_.mat',mydict)
                   counter = counter +1
            elif self._debug and self._update:
                   rvec[:]=rvec_sh[:]
                   shm_rvec.close()
                   shm_rvec.unlink()
                   kappa[:]=kappa_sh[:]
                   shm_kappa.close()
                   shm_kappa.unlink()
                   self._update_threshold(kappa,autocorr)
                   mydict={'rvec':rvec,'kappa':kappa}
                   savemat('gg_'+str(counter)+'_.mat',mydict)
                   counter = counter +1
            elif self._update:
                   rvec[:]=rvec_sh[:]
                   shm_rvec.close()
                   shm_rvec.unlink()
                   kappa[:]=kappa_sh[:]
                   shm_kappa.close()
                   shm_kappa.unlink()
                   self._update_threshold(kappa,autocorr)
            print(f'{flag.sum()/flag.size}')

            for nt in range(self._T):
               t1=iter_w.next()
               tt=t1.query(sortlist='ANTENNA1,ANTENNA2',columns='FLAG')
               if not self._debug:
                   tt.putcol('FLAG',flag_t[nt])

          #########################################
          except:
            print('Finishing up')
            do_iterate=False

        self.close_ms()

    def _probabilities(self,x,kappa_batch):
        # kappa: batch x F, iterate over the batch
        nbatch=kappa_batch.shape[0]
        total_prob=0.0
        # False alarm (constant for kappa)
        Pf=2*qfunc(np.sqrt(3/self._N*x))+np.sqrt(6/(np.pi*self._N)*x)*np.exp(-1.5/self._N*x*x)
        for ci in range(nbatch): 
            kappa=kappa_batch[ci]
            # clamp kappa > 0
            kappa[kappa<self._eps]=self._eps
            # Missed detection
            Pd_numerator=x.flatten()-self._N*(coth(kappa)-1/kappa)-1/kappa
            Pd_denominator=self._N*(1/(kappa**2)-(csch(kappa)**2))-1/(kappa**2)
            Pd_denominator[Pd_denominator<self._eps]=self._eps
            Pd=qfunc(Pd_numerator/np.sqrt(Pd_denominator))
            Ptot=-Pd+Pf
            total_prob+=Ptot.sum()/self._D

        return total_prob/nbatch

    def _probabilities_ratio(self,x,kappa_batch):
        # kappa: batch x F, iterate over the batch
        # ratio=(False alarm prob)/(Detection prob) --> minimize
        nbatch=kappa_batch.shape[0]
        detection_prob=self._eps
        # False alarm (constant for kappa)
        Pf=2*qfunc(np.sqrt(3/self._N*x))+np.sqrt(6/(np.pi*self._N)*x)*np.exp(-1.5/self._N*x*x)
        for ci in range(nbatch): 
            kappa=kappa_batch[ci]
            # clamp kappa > 0
            kappa[kappa<self._eps]=self._eps
            # Missed detection
            Pd_numerator=x.flatten()-self._N*(coth(kappa)-1/kappa)-1/kappa;
            Pd_denominator=self._N*(1/(kappa**2)-(csch(kappa)**2))-1/(kappa**2)
            Pd_denominator[Pd_denominator<self._eps]=self._eps
            Pd=qfunc(Pd_numerator/np.sqrt(Pd_denominator))
            detection_prob+=Pd.sum()

        return Pf.sum()/(detection_prob/nbatch)


    def _probabilities_grad(self,x,kappa_batch):
        # kappa: batch x F, iterate over the batch
        nbatch=kappa_batch.shape[0]
        gradient=np.zeros_like(x)
        x1=3/self._N*x.copy()
        x2=x.copy()
        x1[x1<self._eps]=self._eps
        x2[x2<self._eps]=self._eps

        gPf=2*(-1/np.sqrt(2*np.pi))*np.exp(-x1/2)*0.5*np.sqrt(3/self._N)*np.sqrt(1/x2) \
                + np.sqrt(6/(np.pi*self._N))*(0.5*np.sqrt(1/x2)-3/self._N*x2*np.sqrt(x2))*np.exp(-(3/(2*self._N))*(x**2))

        for ci in range(nbatch): 
            kappa=kappa_batch[ci]
            # clamp kappa > 0
            kappa[kappa<self._eps]=self._eps
            Pd_numerator=x.flatten()-self._N*(coth(kappa)-1/kappa)-1/kappa;
            Pd_denominator=self._N*(1/(kappa**2)-(csch(kappa)**2))-1/(kappa**2)
            Pd_denominator[Pd_denominator<self._eps]=self._eps

            gPd=-1/np.sqrt(2*np.pi)*np.exp(-0.5*(Pd_numerator/np.sqrt(Pd_denominator))**2)/np.sqrt(Pd_denominator)

            gradient=gradient.flatten()+(-gPd.flatten()+gPf.flatten())

        return gradient[:,None]/nbatch


    def _probabilities_ratio_grad(self,x,kappa_batch):
        # kappa: batch x F, iterate over the batch
        # ratio=(False alarm prob)/(Detection prob) =f/g
        # grad=(g f' - f g')/g^2
        nbatch=kappa_batch.shape[0]
        gradient=np.zeros_like(x)
        x1=3/self._N*x.copy()
        x2=x.copy()
        x1[x1<self._eps]=self._eps
        x2[x2<self._eps]=self._eps

        Pf=2*qfunc(np.sqrt(3/self._N*x))+np.sqrt(6/(np.pi*self._N)*x)*np.exp(-1.5/self._N*x*x)
        gPf=2*(-1/np.sqrt(2*np.pi))*np.exp(-x1/2)*0.5*np.sqrt(3/self._N)*np.sqrt(1/x2) \
                + np.sqrt(6/(np.pi*self._N))*(0.5*np.sqrt(1/x2)-3/self._N*x2*np.sqrt(x2))*np.exp(-(3/(2*self._N))*(x**2))

        for ci in range(nbatch): 
            kappa=kappa_batch[ci]
            # clamp kappa > 0
            kappa[kappa<self._eps]=self._eps
            Pd_numerator=x.flatten()-self._N*(coth(kappa)-1/kappa)-1/kappa;
            Pd_denominator=self._N*(1/(kappa**2)-(csch(kappa)**2))-1/(kappa**2)
            Pd_denominator[Pd_denominator<self._eps]=self._eps

            Pd=qfunc(Pd_numerator/np.sqrt(Pd_denominator)).sum()
            gPd=-1/np.sqrt(2*np.pi)*np.exp(-0.5*(Pd_numerator/np.sqrt(Pd_denominator))**2)/np.sqrt(Pd_denominator)

            gradient=gradient.flatten()+(Pd*gPf.flatten()-Pf.sum()*gPd.flatten())/(Pd*Pd+self._eps)

        return gradient[:,None]/nbatch


    def _grid_search(self,cost):
        # perform a grid search
        Ngrid=200
        Delta=0.1
        x=np.ones(self._D)*self._g_low*0.01
        costs=np.zeros(Ngrid)
        for ci in range(Ngrid):
            costs[ci]=cost(x+Delta*ci)
            #print(f'x={x[0]+Delta*ci} cost={costs[ci]}')
        minval=np.argmin(costs)
        return x+Delta*minval

    def _update_threshold(self,Kappa,autocorr):
        # Kappa: _B x _D x 2
        # autocorr: _B array, True for autocorrelations
        # randomly select a baseline
        batch_size=100

        for admm in range(self._ADMM):
           bline=np.random.choice(self._B,batch_size,replace=False)
           # remove autocorrelations
           bline=bline[np.logical_not(autocorr[bline])]
           re_im=np.random.choice(2,1,replace=False)

           def cost_func(x):
               x_Bz=(x-self._cB.transpose() @ self._z)
               cost=self._probabilities(x,Kappa[bline,:,re_im]) + np.sum(self._y*x_Bz + 0.5*self._rho*(x_Bz**2))
               return cost

           def grad_func(x):
               x_Bz=(x-self._cB.transpose() @ self._z)
               gradient=self._probabilities_grad(x,Kappa[bline,:,re_im]) + self._y + self._rho*x_Bz
               return gradient

           #self._solver.grad_check(cost_func,grad_func)
           self._solver.step(cost_func,grad_func)
           self._g=self._solver.x.squeeze().copy()
           #self._g=self._grid_search(cost_func)

           lhs=(self._y.squeeze()+self._rho*self._g) @ self._cB.transpose()
           znew=self._cBi @ lhs
           dual_res=np.linalg.norm(self._z-znew[:,None])
           self._z=znew[:,None]

           y=self._y+self._rho*(self._g[:,None]-self._cB.transpose() @ self._z)
           primal_res=np.linalg.norm(self._y-y)
           self._y=y
           #print(f'{admm} {primal_res} {dual_res}')


    def flag_spectral_kurtosis(self,X):
        if self._finite_prec:
          return  self.flag_spectral_kurtosis_fp_(X)
        else: 
          return  self.flag_spectral_kurtosis_(X)

    def flag_spectral_kurtosis_fp_(self,X):
        # X: TxFx1 data (complex)  - cross correlations, not voltages
        # refer Nita&Helb URSI 2020
        # return: flags: Fx4 bool, 4 pol
        # Note: we override the window size self._W
        # and use single channel as the window size

        M=X.shape[0] # averaging length (in time)
        F=X.shape[1] # channels


        TYPE=torch.float32 if self._finite_act_sk[0]==0 else torch.float16
        Xr=torch.tensor(X.real).to(TYPE)
        Xi=torch.tensor(X.imag).to(TYPE)
        X2=Xr**2+Xi**2

        TYPE=torch.float32 if self._finite_act_sk[1]==0 else torch.float16
        Xa=torch.sqrt(X2).to(TYPE)

        TYPE=torch.float32 if self._finite_act_sk[2]==0 else torch.float16
        S1=torch.sum(Xa,axis=0).to(TYPE)
        
        TYPE=torch.float32 if self._finite_act_sk[3]==0 else torch.float16
        S2=torch.sum(X2,axis=0).to(TYPE)

        TYPE=torch.float32 if self._finite_act_sk[4]==0 else torch.float16
        # generalized spectral kurtosis
        t=torch.tensor((M*self._sk_d+1)/(M-1)).to(TYPE)

        TYPE=torch.float32 if self._finite_act_sk[5]==0 else torch.float16
        S12=(S1*S1).to(TYPE)

        TYPE=torch.float32 if self._finite_act_sk[6]==0 else torch.float16
        t = t*(M*S2/(S12+self._eps)-1.0).to(TYPE)

        fl=(t < self._sk_low) | (t > self._sk_high)

        flags=np.zeros((F,4),dtype=bool)
        for ci in range(4):
            flags[:,ci]=fl.numpy()

        return flags


    def flag_spectral_kurtosis_(self,X):
        # X: TxFx1 data (complex)  - cross correlations, not voltages
        # refer Nita&Helb URSI 2020
        # return: flags: Fx4 bool, 4 pol
        # Note: we override the window size self._W
        # and use single channel as the window size
        M=X.shape[0] # averaging length (in time)
        F=X.shape[1] # channels
        S1=np.sum(np.abs(X),axis=0)
        S2=np.sum(np.abs(X)**2,axis=0)
        # generalized spectral kurtosis
        t=((M*self._sk_d+1)/(M-1))*(M*S2/(S1*S1+self._eps)-1.0)
        fl=(t < self._sk_low) | (t > self._sk_high)
        flags=np.zeros((F,4),dtype=bool)
        for ci in range(4):
            flags[:,ci]=fl

        return flags
        
    def info(self):
        print(f'MS {self._ms} has {self._Nant} ant with {self._F} freqs');


if __name__=='__main__':
    parser=argparse.ArgumentParser(
      description='Energy and polarization based flagging',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--MS',type=str,metavar='\'*.MS\'',
        help='MS to use')
    parser.add_argument('--col',type=str,default='DATA',metavar='d',
        help='Column name DATA|MODEL_DATA|...')
    parser.add_argument("--method", default="BOTH", help='method to use: DS (directional statistics), SK (spectral kurtosis), BOTH, SK4 (spectral kurtosis for each polarization)')  # select RFI flagging method
    parser.add_argument('--preserve', default=False, action=argparse.BooleanOptionalAction, help='Preserve old flags')
    parser.add_argument('--debug', default=False, action=argparse.BooleanOptionalAction, help='Enable debugging in sequential mode')
    parser.add_argument('--parallel_jobs',type=int,default=4,metavar='p',
            help='Number of parallel jobs')
    parser.add_argument('--admm_rho',type=float,default=0.01,metavar='r',
            help='ADMM regularization')
    parser.add_argument('--admm_iterations',type=int,default=10,metavar='a',
            help='Number of ADMM iterations')
    parser.add_argument('--freq_window_size',type=int,default=1,metavar='w',
            help='Window size (in channels) to use for RFI detection')
    parser.add_argument('--time_window_size',type=int,default=10,metavar='t',
            help='Window size (in time) to use for RFI detection')
    parser.add_argument('--update', default=False, action=argparse.BooleanOptionalAction, help='Update threshould by online optimization')
    parser.add_argument('--finite_prec', default=False, action=argparse.BooleanOptionalAction, help='If true, enable finite precision evalution')

    args=parser.parse_args()

    if args.MS:
        pf=PolFlagger(args.MS,args.method,args.col,args.preserve,do_debug=args.debug,do_update=args.update,parallel_jobs=args.parallel_jobs,freq_window_size=args.freq_window_size,time_window_size=args.time_window_size,admm_rho=args.admm_rho,admm_iter=args.admm_iterations,finite_prec=args.finite_prec)
        pf.iter_ms()
    else:
        parser.print_help()
