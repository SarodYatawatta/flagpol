import numpy as np
import matplotlib.pyplot as plt
from lofarvis import LOFARvis

# try to import external modules
have_extended=False
try:
    from astropy.units import sday
    from hera_sim import defaults, foregrounds, noise

    from pyphysim.modulators.fundamental import BPSK, QAM, QPSK, Modulator
    from pyphysim.modulators import OFDM
    from pyphysim import channels
    from pyphysim.channels.fading import COST259_RAx, TdlChannel
    from pyphysim.channels.fading_generators import JakesSampleGenerator
    from pyphysim.util.misc import randn_c
    have_extended=True 
except ImportError:
    print('Cannot import external modules, simulation with limited capabilities')


class Simulator:
    def __init__(self,n_time=1e6,n_freq=512,extended_simul=False):
        # full window size in time x frequency samples
        self.n_time=int(n_time)
        self.n_freq=int(n_freq)
        # full window domain
        self.T=1.08 # seconds
        self.F=200e3 # Hz
        # reference frequency
        self.f0=120e6 # Hz
        # enable extended simulation?
        self.extended_simul=have_extended and extended_simul 

        # number of celestial sources
        self.n_src=100
        # number of RFI sources
        self.n_rfi=10 #10

        if self.extended_simul:
            self.lofarvis=LOFARvis(n_time=self.n_time,n_freq=self.n_freq,n_rfi=2)
        else:
            self.lofarvis=None

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
        """
        At the end: XX,XY,YX,YY will have sky+noise
        XX_rfi,XY_rfi,YX_rfi,YY_rif will have RFI
        """
        # power law sky noise
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

        # add receiver noise, first simulate two voltage bins
        recv_noise_scale=10
        V_1x=np.random.randn(self.n_time,self.n_freq)+1j*np.random.randn(self.n_time,self.n_freq)
        V_1y=np.random.randn(self.n_time,self.n_freq)+1j*np.random.randn(self.n_time,self.n_freq)
        V_2x=np.random.randn(self.n_time,self.n_freq)+1j*np.random.randn(self.n_time,self.n_freq)
        V_2y=np.random.randn(self.n_time,self.n_freq)+1j*np.random.randn(self.n_time,self.n_freq)
        self.XX +=recv_noise_scale*V_1x*np.conj(V_2x)
        self.XY +=recv_noise_scale*V_1x*np.conj(V_2y)
        self.YX +=recv_noise_scale*V_1y*np.conj(V_2x)
        self.YY +=recv_noise_scale*V_1y*np.conj(V_2y)

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

    def generate_data_fx(self):
        """
        FX correlator emulation, using HERA_sim and pyphysim
        At the end: XX,XY,YX,YY will have sky+noise
        XX_rfi,XY_rfi,YX_rfi,YY_rif will have RFI
        """

        # simulate RFI sources (all have unit amplitude)
        self.XX,self.XY,self.YX,self.YY, \
                  self.XX_rfi, self.XY_rfi, self.YX_rfi, self.YY_rfi=self.lofarvis.generate_data()

        # update n_time from generated data
        self.n_time=self.XX.shape[0]
        # update n_freq (more than one subband)
        n_subbands=self.XX.shape[1]//self.n_freq

        # HERA specific stuff
        defaults.set("h1c")
        # frequencies are in GHz, LST: 0~2pi angle
        flow=np.random.randint(30,40)/1e3 # GHz
        fqs = np.linspace(flow, flow+n_subbands*self.F/1e9, self.n_freq*n_subbands, endpoint=False)
        philow=np.random.rand()*np.pi
        lsts = np.linspace(philow, philow+2*np.pi/sday.to('s')*self.T, self.n_time, endpoint=False)
        bl_len_ns = np.array([np.random.rand()*100.0, np.random.rand()*100, np.random.rand()*10])
        h1c_beam = defaults("omega_p")(fqs)
        Tsky_mdl = noise.HERA_Tsky_mdl["xx"]
        vis_fg_diffuse = foregrounds.diffuse_foreground(
          lsts, fqs, bl_len_ns, Tsky_mdl=Tsky_mdl, omega_p=h1c_beam
        )
        vis_fg_pntsrc = foregrounds.pntsrc_foreground(lsts, fqs, bl_len_ns, nsrcs=self.n_src)
        vis_fg = vis_fg_diffuse + vis_fg_pntsrc
        # sky polarization ~ 0.1%
        nos_jy = noise.sky_noise_jy(
           lsts, fqs, omega_p=h1c_beam, Tsky_mdl=noise.HERA_Tsky_mdl['xx']
        )
        self.XX += vis_fg +nos_jy
        nos_jy = noise.sky_noise_jy(
           lsts, fqs, omega_p=h1c_beam, Tsky_mdl=noise.HERA_Tsky_mdl['xx']
        )
        self.XY += 0.001*vis_fg +nos_jy
        nos_jy = noise.sky_noise_jy(
           lsts, fqs, omega_p=h1c_beam, Tsky_mdl=noise.HERA_Tsky_mdl['yy']
        )
        self.YX += 0.001*vis_fg +nos_jy
        nos_jy = noise.sky_noise_jy(
           lsts, fqs, omega_p=h1c_beam, Tsky_mdl=noise.HERA_Tsky_mdl['yy']
        )
        self.YY += vis_fg +nos_jy


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

#sim=Simulator(n_time=2000,n_freq=512,extended_simul=True)
#sim.generate_data()
#sim.generate_data_fx()
#sim.plot_data()
