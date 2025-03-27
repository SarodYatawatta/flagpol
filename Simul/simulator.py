import numpy as np
import matplotlib.pyplot as plt

class Simulator:
    def __init__(self,n_time=1e6,n_freq=512):
        # full window size in time x frequency samples
        self.n_time=int(n_time)
        self.n_freq=int(n_freq)
        # full window domain
        self.T=1.08 # seconds
        self.F=200e3 # Hz
        # reference frequency
        self.f0=120e6 # Hz

        # number of celestial sources
        self.n_src=100
        # number of RFI sources
        self.n_rfi=10
        # frequencies
        self.f=np.linspace(self.f0-0.5*self.F/self.n_freq,self.f0+0.5*self.F/self.n_freq,num=self.n_freq)
        # times (normalized)
        self.t=np.linspace(0,self.T,num=self.n_time)

        # scale factors
        self.max_rfi_scale=10000
        self.max_sky_scale=10
        # probability to add RFI, if U(0,1) > this value
        self.rfi_prob=0.2

        self.XX=None
        self.XY=None
        self.YX=None
        self.YY=None
        self.XX_rfi=None
        self.XY_rfi=None
        self.YX_rfi=None
        self.YY_rfi=None

    def generate_data(self):
        # power law noise
        skynoise=(self.f/self.f0)**(-2.7/2)
        self.XX=np.random.randn(self.n_time,self.n_freq)+1j*np.random.randn(self.n_time,self.n_freq)
        self.XY=np.random.randn(self.n_time,self.n_freq)+1j*np.random.randn(self.n_time,self.n_freq)
        self.YX=np.random.randn(self.n_time,self.n_freq)+1j*np.random.randn(self.n_time,self.n_freq)
        self.YY=np.random.randn(self.n_time,self.n_freq)+1j*np.random.randn(self.n_time,self.n_freq)
        for t in range(self.n_time):
            self.XX[t,:] = self.XX[t,:]*skynoise
            self.XY[t,:] = self.XY[t,:]*skynoise
            self.YX[t,:] = self.YX[t,:]*skynoise
            self.YY[t,:] = self.YY[t,:]*skynoise

        # simulate celestial sources
        for ci in range(self.n_src):
            # ul+vm+w(n-1) over time domain
            lmn=np.random.randn(1)*np.cos(self.t/(np.random.rand(1)*20))
            # 2 pi f/c over freq, converted to MHz
            pifc=2*np.pi*self.f/(3e8)*1e6
            sexp=np.exp(-1j*np.outer(lmn,pifc))
            iquv=np.random.rand(4)*self.max_sky_scale
            self.XX +=sexp*iquv[0]
            self.XY +=sexp*iquv[1]
            self.YX +=sexp*iquv[2]
            self.YY +=sexp*iquv[3]

        self.XX_rfi=np.zeros((self.n_time,self.n_freq))+1j*np.zeros((self.n_time,self.n_freq))
        self.XY_rfi=np.zeros((self.n_time,self.n_freq))+1j*np.zeros((self.n_time,self.n_freq))
        self.YX_rfi=np.zeros((self.n_time,self.n_freq))+1j*np.zeros((self.n_time,self.n_freq))
        self.YY_rfi=np.zeros((self.n_time,self.n_freq))+1j*np.zeros((self.n_time,self.n_freq))
        # simulate RFI sources (all have unit amplitude)
        if np.random.rand() > self.rfi_prob:
          for ci in range(self.n_rfi):
              # two types: 0 or 1
              stype=np.random.randint(0,2)
              J=(np.random.randn(4)+1j*np.random.randn(4))
              #for unpolarized RFI, uncomment following
              #J[3]=J[0]
              #J[1]=0
              #J[2]=0
              scalefactor=1/np.linalg.norm(J)
              if stype==0:
                 t_low=np.random.randint(0,self.n_time)
                 t_high=np.random.randint(t_low,self.n_time)
                 f_low=np.random.randint(0,self.n_freq)
                 f_high=np.random.randint(f_low,self.n_freq)
                 t_w=t_high-t_low+1
                 f_w=f_high-f_low+1
                 rfi=np.ones((t_w,f_w))
                 self.XX_rfi[t_low:t_high+1,f_low:f_high+1] +=rfi*J[0]
                 self.XY_rfi[t_low:t_high+1,f_low:f_high+1] +=rfi*J[1]
                 self.YX_rfi[t_low:t_high+1,f_low:f_high+1] +=rfi*J[2]
                 self.YY_rfi[t_low:t_high+1,f_low:f_high+1] +=rfi*J[3]
              else:
                 # time or freq?
                 stype=np.random.randint(0,2)
                 if stype==0:
                    # carrier freq
                    fi=np.random.randint(0,self.n_freq)
                    # period
                    period=np.random.rand(1)*self.n_time
                    li=np.exp(1j*(self.t/period))
                    self.XX_rfi[:,fi] +=li*J[0]
                    self.XY_rfi[:,fi] +=li*J[1]
                    self.YX_rfi[:,fi] +=li*J[2]
                    self.YY_rfi[:,fi] +=li*J[3]
                 else:
                    # carrier time
                    fi=np.random.randint(0,self.n_time)
                    # period
                    period=np.random.rand(1)*self.n_freq*self.f0
                    li=np.exp(1j*(self.f/period))
                    self.XX_rfi[fi,:] +=li*J[0]
                    self.XY_rfi[fi,:] +=li*J[1]
                    self.YX_rfi[fi,:] +=li*J[2]
                    self.YY_rfi[fi,:] +=li*J[3]


    def plot_data(self,SNR=100,filename='bar.png'):
       fig=plt.figure(1)

       fig,axs=plt.subplots(1,4)
       im=axs[0].imshow(np.log(np.abs(self.XX+self.XX_rfi*SNR)))
       cbar=plt.colorbar(im,ax=axs[0])
       cbar.ax.tick_params(labelsize=5)
       im.axes.get_xaxis().set_ticks([])
       im.axes.get_yaxis().set_ticks([])
       im=axs[1].imshow(np.log(np.abs(self.XY+self.XY_rfi*SNR)))
       cbar=plt.colorbar(im,ax=axs[1])
       cbar.ax.tick_params(labelsize=5)
       im.axes.get_xaxis().set_ticks([])
       im.axes.get_yaxis().set_ticks([])
       im=axs[2].imshow(np.log(np.abs(self.YX+self.YX_rfi*SNR)))
       cbar=plt.colorbar(im,ax=axs[2])
       cbar.ax.tick_params(labelsize=5)
       im.axes.get_xaxis().set_ticks([])
       im.axes.get_yaxis().set_ticks([])
       im=axs[3].imshow(np.log(np.abs(self.YY+self.YY_rfi*SNR)))
       cbar=plt.colorbar(im,ax=axs[3])
       cbar.ax.tick_params(labelsize=5)
       im.axes.get_xaxis().set_ticks([])
       im.axes.get_yaxis().set_ticks([])
       axs[0].set(xlabel='Freq',ylabel='Time', title='XX')
       axs[1].set(xlabel='Freq',ylabel='', title='XY')
       axs[2].set(xlabel='Freq',ylabel='', title='YX')
       axs[3].set(xlabel='Freq',ylabel='', title='YY')
       plt.savefig(filename)

    def reset(self):
        self.XX=None
        self.XY=None
        self.YX=None
        self.YY=None
        self.XX_rfi=None
        self.XY_rfi=None
        self.YX_rfi=None
        self.YY_rfi=None

#sim=Simulator(n_time=2000,n_freq=512)
#sim.generate_data()
#sim.plot_data()
