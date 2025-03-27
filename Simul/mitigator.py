import numpy as np
from scipy import special as sp
import matplotlib.pyplot as plt
from simulator import Simulator

EPS=1e-6

def invQfunc(x):
    return np.sqrt(2)*sp.erfinv(1-2*x)

def ds_lower_bound(N,Pf):
   P_1=1.856697
   P_2=0.283628;
   # formula is valid for 20 < N < 1000
   g_low=np.sqrt(N/(3*P_2)*invQfunc(Pf/P_1))

   return g_low

# spectral kurtosis upper/lower bounds
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


class Mitigator:
    def __init__(self,n_time=1e6,n_freq=512,w_time=10, w_freq=2):
        # full window size in time x frequency samples
        self.n_time=int(n_time)
        self.n_freq=int(n_freq)
        # RFI mitigation window size
        self.w_time=int(w_time)
        self.w_freq=int(w_freq)

        self.sim=Simulator(n_time=self.n_time,n_freq=self.n_freq)

        # False alarm prob
        self.Pf=0.01
        # SK d
        self.sk_d=1.0/2
        # thresholds
        self.SK_low,self.SK_high=gsk_bounds(self.w_time*self.w_freq,self.sk_d,self.Pf)
        self.SK_low=0.05
        self.SK_high=2.2
        self.DS_gamma=ds_lower_bound(self.w_time*self.w_freq,self.Pf)*1.5
        print(f'Thresholds for window {self.w_time*self.w_freq} gamma {self.DS_gamma} SK {self.SK_low} {self.SK_high}')

    def kernel_spectral_kurtosis(self,Xr,Xi):
        # X: complex (cross correlations, Stokes I), T x 1 vector, Xr,Xi: real/imaginary parts
        # return flags: 1 x 1
        M=Xr.shape[0] # averaging length
        X2=Xr*Xr+Xi*Xi # st 1
        # find abs(X)
        Xa=np.sqrt(X2) # st 2
        S1=np.sum(Xa) # st 3
        X4=X2 # == Xa**2 # st 4
        S2=np.sum(X4) # st 5
        SK_d=self.sk_d  # st 6 - constants
        SK_low=self.SK_low # st 6
        SK_high=self.SK_high # st 6
        S12=S1*S1+EPS # st 7
        t=((M*SK_d+1)/(M-1))*(M*S2/(S12)-1) # st 8
        fll=(t<SK_low)
        fhh=(t>SK_high)
        flags=fll | fhh

        return flags


    def kernel_dir_statistics(self,Q,U,V):
        # Q,U,V: real, T x 1 vectors
        # return flags 1x1
        N=Q.shape[0] # averaging length
        Q2=Q**2 # st 1
        U2=U**2 # st 1
        V2=V**2 # st 1
        pp=np.sqrt(Q2+U2+V2) # st 2
        Q=Q/pp # st 3
        U=U/pp # st 3
        V=V/pp # st 3
        # sum whole block into scalar
        qq=np.sum(Q) # st 4
        uu=np.sum(U) # st 4
        vv=np.sum(V) # st 4
        qq2=qq**2 # st 5
        uu2=uu**2 # st 5
        vv2=vv**2 # st 5
        rr=qq2+uu2+vv2 # st 6
        rvec=np.sqrt(rr)/N # st 6
        DS_gamma=self.DS_gamma # st 7 - constants
        flags=rvec > DS_gamma/N

        return flags
    
    def run_mitigation(self,SNR):
        """
         return False alarm, missed detections
         for both methods
        """
        if self.sim.XX is None:
           self.sim.generate_data()

        # SNR: signal/noise, signal=RFI, noise=background
        XX=self.sim.XX+SNR*self.sim.XX_rfi
        XY=self.sim.XY+SNR*self.sim.XY_rfi
        YX=self.sim.YX+SNR*self.sim.YX_rfi
        YY=self.sim.YY+SNR*self.sim.YY_rfi

        #self.plot_data(XX,XY,YX,YY,'before.png')
        Iw=XX+YY
        Qw=XX-YY
        Uw=XY+YX
        Vw=1j*(XY-YX)

        # ground truth RFI mask
        m_rfi=(np.abs(self.sim.XX_rfi)>0) | (np.abs(self.sim.XY_rfi)>0) | (np.abs(self.sim.YX_rfi)>0) | (np.abs(self.sim.YY_rfi)>0)

        # RFI masks
        m_ds=np.zeros(XX.shape,dtype=bool)
        m_sk=np.zeros(XX.shape,dtype=bool)

        n_f=self.n_freq//self.w_freq
        n_t=self.n_time//self.w_time
        # iterate over time,freq windows
        for t in range(n_t):
            for f in range(n_f):
                I=Iw[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq].flatten()
                Q=Qw[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq].flatten()

                U=Uw[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq].flatten()

                V=Vw[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq].flatten()

                fl=self.kernel_spectral_kurtosis(np.real(I),np.imag(I))
                fl1=self.kernel_dir_statistics(np.real(Q),np.real(U),np.real(V))
                fl2=self.kernel_dir_statistics(np.imag(Q),np.imag(U),np.imag(V))
                if fl:
                   m_sk[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq] = 1
                if fl1 | fl2:
                   m_ds[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq] = 1
                if fl|fl1|fl2:
                   XX[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq]=EPS
                   XY[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq]=EPS
                   YX[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq]=EPS
                   YY[t*self.w_time:(t+1)*self.w_time,f*self.w_freq:(f+1)*self.w_freq]=EPS

        #self.plot_data(XX,XY,YX,YY,'after.png')
        sk_miss,sk_false,ds_miss,ds_false,both_miss,both_false=self.calculate_probabilities(m_rfi,m_sk,m_ds)
        print(f'{SNR} {sk_miss} {sk_false} {ds_miss} {ds_false} {both_miss} {both_false}')

    def calculate_probabilities(self,mask_0,mask_sk,mask_ds):
        """
        ground truth is mask_0
        return false alarm, missed detection for SK,DS,BOTH
        """
        badcount=mask_0.sum()

        mask_both=mask_sk | mask_ds

        # correct detections
        m_corr=mask_0 & mask_sk
        # missed detections
        if badcount > 0:
          missed_sk=(badcount- m_corr.sum())/badcount
        else:
          missed_sk=(m_corr.sum())/mask_0.size
        # correct detections
        m_corr=mask_0 & mask_ds
        # missed detections
        if badcount > 0:
          missed_ds=(badcount- m_corr.sum())/badcount
        else:
          missed_ds=(m_corr.sum())/mask_0.size
        # correct detections
        m_corr=mask_0 & mask_both
        # missed detections
        if badcount > 0:
          missed_both=(badcount- m_corr.sum())/badcount
        else:
          missed_both=(m_corr.sum())/mask_0.size

        # false detections
        goodcount=mask_0.size - badcount
        if goodcount==0:
            goodcount=1
        false_sk=((~mask_0) & mask_sk).sum()/goodcount
        false_ds=((~mask_0) & mask_ds).sum()/goodcount
        false_both=((~mask_0) & mask_both).sum()/goodcount

        return missed_sk,false_sk,missed_ds,false_ds,missed_both,false_both

    def plot_data(self,XX,XY,YX,YY,filename='foo.png'):
       font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
       }

       fig=plt.figure(1)
       fig.tight_layout()

       fig,axs=plt.subplots(1,4)
       im=axs[0].imshow(np.log(np.abs(XX)))
       #cbar=plt.colorbar(im,ax=axs[0])
       #cbar.ax.tick_params(labelsize=5)
       im.axes.get_xaxis().set_ticks([])
       im.axes.get_yaxis().set_ticks([])
       im=axs[1].imshow(np.log(np.abs(XY)))
       #cbar=plt.colorbar(im,ax=axs[1])
       #cbar.ax.tick_params(labelsize=5)
       im.axes.get_xaxis().set_ticks([])
       im.axes.get_yaxis().set_ticks([])
       im=axs[2].imshow(np.log(np.abs(YX)))
       #cbar=plt.colorbar(im,ax=axs[2])
       #cbar.ax.tick_params(labelsize=5)
       im.axes.get_xaxis().set_ticks([])
       im.axes.get_yaxis().set_ticks([])
       im=axs[3].imshow(np.log(np.abs(YY)))
       maxval=np.max(np.log(np.abs(YY)))
       minval=np.min(np.log(np.abs(YY)))
       cbar=plt.colorbar(im,ax=axs[3],ticks=[-10, 0, 5])
       cbar.ax.tick_params(labelsize=6)
       cbar.set_label('Log amplitude')
       im.axes.get_xaxis().set_ticks([])
       im.axes.get_yaxis().set_ticks([])
       axs[0].set(xlabel='Frequency',ylabel='Time', title='XX')
       axs[1].set(xlabel='Frequency',ylabel='', title='XY')
       axs[2].set(xlabel='Frequency',ylabel='', title='YX')
       axs[3].set(xlabel='Frequency',ylabel='', title='YY')
       plt.tight_layout()
       plt.savefig(filename)

    def reset(self):
        self.sim.reset()

    def run_loop(self,SNRs):
        for SNR in SNRs:
            self.run_mitigation(SNR)

mit=Mitigator(n_time=2000,n_freq=512,w_time=10,w_freq=2)
#mit.run_mitigation(100)

# loop over simulations
for nsim in range(400):
   mit.reset()
   mit.run_loop(SNRs=[1, 5, 10, 15, 20, 30, 50, 100, 200, 300, 500])
