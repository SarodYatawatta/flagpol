import math
import numpy as np
import scipy
from scipy.signal import firwin

import matplotlib.pyplot as plt

# try to import external modules
have_extended=False
try:
    from pyphysim.modulators.fundamental import BPSK, QAM, QPSK, Modulator
    from pyphysim.modulators import OFDM
    from pyphysim import channels
    from pyphysim.channels.fading import COST259_RAx, TdlChannel
    from pyphysim.channels.fading_generators import JakesSampleGenerator
    from pyphysim.util.misc import randn_c
    have_extended=True
except ImportError:
    print('Cannot import external modules, simulation with limited capabilities')


class LOFARvis:
    """
    Simulate signal path from two stations to correlator:
     1) subband polyphase filterbank
     2) channelize voltae data for a selected (some) subbands
     3) perform product of channelized data at correlator

    Signals : RFI + receiver noise
    """
    def __init__(self,n_time=1e6,n_freq=512,n_rfi=10):
        # Note n_time,n_freq below are the window for 
        # each subband
        # and n_time will not always match the generated data size
        self.n_time=int(n_time) 
        self.n_freq=int(n_freq)
        self.n_rfi=n_rfi
        # full window domain
        self.T=1.08 # seconds
        self.F=200e3 # Hz
        self.f_clock=200e6 # sampling clock
        # selected subbands
        self.subbands=np.arange(200,204)
        # selected subband frequencies
        self.subband_freqs=self.subbands*0.2e6

        # number of polyphase components (subbands)
        self.n_poly=1024 # 1024x200kHz ~ 200MHz
        # length of each polyphase filter
        self.n_taps=16

        

    # helper functions for polyphase channelizer
    # Based on the code from Evan Smith
    ## https://github.com/etsmit/RFI_Simulations
    #x: input 1d data stream
    #win_coeffs: window coefficients (from??)
    #n_taps: # of taps used by polyphase filter
    #n_poly: # of polyphase components (=channels)
    def pfb_fir_frontend_(self,x, win_coeffs, n_taps, n_poly):
        W = int(x.shape[0] / n_taps / n_poly)
        x_p = x[:W*n_taps*n_poly].reshape((W*n_taps, n_poly)).T
        h_p = win_coeffs.reshape((n_taps, n_poly)).T
        x_summed = np.zeros((n_poly, n_taps * W - n_taps),dtype=np.complex64)
        for t in range(0, n_taps*W-n_taps):
            x_weighted = x_p[:, t:t+n_taps] * h_p
            x_summed[:, t] = x_weighted.sum(axis=1)
        return x_summed.T

    def generate_win_coeffs_(self,n_taps, n_poly, window_fn="hann"):
        win_coeffs = scipy.signal.get_window(window_fn, n_taps*n_poly)
        sinc       = scipy.signal.firwin(n_taps * n_poly, cutoff=1.0/float(n_poly), window="rectangular")
        win_coeffs *= sinc
        return win_coeffs

    def pfb_filterbank_(self,x, n_taps, n_poly, window_fn="hann"):
        """
        x: time series complex data, channelized to
        n_poly channels, using FIR filter with n_taps in each polyphase component
        """
        win_coeffs = self.generate_win_coeffs_(n_taps, n_poly, window_fn)
        x_fir = self.pfb_fir_frontend_(x, win_coeffs, n_taps, n_poly)
        x_pfb = np.fft.fft(x_fir, axis=1)
        return x_pfb

    def raised_cosine_(self,n_sps=8,n_taps=101,beta=0.35):
        """
        raised cosine pulse shaping
        """
        t=np.arange(n_taps)-(n_taps-1)//2
        h=np.ones(t.shape)
        g=1-(2*beta*t/n_sps)**2
        I=np.abs(g)>0
        h[I]=np.sinc(t[I]/n_sps)*np.cos(np.pi*beta*t[I]/n_sps)/(1-(2*beta*t[I]/n_sps)**2)
        return h

    def channelize_(self,data):
        y=self.pfb_filterbank_(data,n_taps=self.n_taps,n_poly=self.n_poly)
        total_time=y.shape[0]
        n_time1=total_time//self.n_freq
        n_subbands=self.subbands.size
        XX=np.zeros((n_time1,n_subbands*self.n_freq))+1j*np.zeros((n_time1,n_subbands*self.n_freq))
        ci=0
        for sb in self.subbands:
            xx=y[:n_time1*self.n_freq,sb].reshape(n_time1,self.n_freq)
            yy=np.fft.fftshift(np.fft.fft(xx,axis=1),axes=(1,))
            XX[:,ci*self.n_freq:(ci+1)*self.n_freq]=yy
            ci +=1

        return XX

    def generate_data(self):
        n_time=int(self.T*self.f_clock)
        # full time range
        t=np.linspace(0,self.T,num=n_time)

        data_1x=np.zeros(n_time)+1j*np.zeros(n_time)
        data_1y=np.zeros(n_time)+1j*np.zeros(n_time)
        data_2x=np.zeros(n_time)+1j*np.zeros(n_time)
        data_2y=np.zeros(n_time)+1j*np.zeros(n_time)

        for ns in range(self.n_rfi):
            # select subset of time < 1
            t_low=np.random.randint(0,n_time)
            t_high=np.random.randint(t_low,n_time)
            t1=t[t_low:t_high+1]
            T=t1[-1]-t1[0]

            # simulate comm signal with low symbol rates ~ 1~10 MHz
            symbol_rate=np.random.randint(1,11)*1e6
            # carrier frequency ~ within the chosen subband freqs
            carrier_freq=self.subband_freqs[np.random.choice(self.subbands.size)]+np.random.rand()*1e6
            n_symbols=int(T*symbol_rate)
            # samples per one symbol
            n_sps=int((t_high-t_low)/n_symbols)
            stype=np.random.randint(0,2)
            if stype==0:
               qpsk=QPSK()
               data_qpsk=np.random.randint(0,qpsk.M,size=n_symbols)
               data=qpsk.modulate(data_qpsk)
            else:
               M=16
               qam=QAM(M)
               data_qam=np.random.randint(0,M,n_symbols)
               data=qam.modulate(data_qam)

            #print(f'samples per symb {n_sps} at rate {symbol_rate/1e6} MHz carrier {carrier_freq/1e6} MHz')
            data_ov=np.zeros(n_sps*n_symbols)+1j*np.zeros(n_sps*n_symbols)
            data_ov[::n_sps]=data
            rc=self.raised_cosine_(n_sps=n_sps,n_taps=101,beta=0.3)
            data1=np.convolve(data_ov,rc)
            # channel (just 2x2 scalar polarizations)
            J=(np.random.randn(4)+1j*np.random.randn(4))

            # add carrier
            carrier=np.exp(1j*np.linspace(t1[0],t1[-1],num=data1.size)*2*np.pi*carrier_freq)
            data1 *=carrier

            data_1x[t_low:t_low+data1.size]+=data1*J[0]
            data_1y[t_low:t_low+data1.size]+=data1*J[1]
            data_2x[t_low:t_low+data1.size]+=data1*J[2]
            data_2y[t_low:t_low+data1.size]+=data1*J[3]

            del data,data1,data_ov,carrier

        # pass through channelizer
        V1X=self.channelize_(data_1x)
        V1Y=self.channelize_(data_1y)
        V2X=self.channelize_(data_2x)
        V2Y=self.channelize_(data_2y)

        del data_1x,data_1y,data_2x,data_2y

        # noise
        noise_var=1e-3
        noise_1x=math.sqrt(noise_var)*randn_c(n_time)
        noise_1y=math.sqrt(noise_var)*randn_c(n_time)
        noise_2x=math.sqrt(noise_var)*randn_c(n_time)
        noise_2y=math.sqrt(noise_var)*randn_c(n_time)


        V1X_noise=self.channelize_(noise_1x)
        V1Y_noise=self.channelize_(noise_1y)
        V2X_noise=self.channelize_(noise_2x)
        V2Y_noise=self.channelize_(noise_2y)

        del noise_1x,noise_1y,noise_2x,noise_2y
            
        XX=V1X*np.conj(V2X)
        XY=V1X*np.conj(V2Y)
        YX=V1Y*np.conj(V2X)
        YY=V1Y*np.conj(V2Y)

        del V1X,V1Y,V2X,V2Y

        XX_noise=V1X_noise*np.conj(V2X_noise)
        XY_noise=V1X_noise*np.conj(V2Y_noise)
        YX_noise=V1Y_noise*np.conj(V2X_noise)
        YY_noise=V1Y_noise*np.conj(V2Y_noise)

        del V1X_noise,V1Y_noise,V2X_noise,V2Y_noise

        #fig=plt.figure(1)
        #plt.imshow(np.abs(XX),aspect='auto',interpolation='none')
        #plt.xlabel('subband')
        #plt.ylabel('time')
        #plt.colorbar()
        #plt.savefig('xx.png')

        #fig=plt.figure(2)
        #plt.imshow(np.abs(XY),aspect='auto',interpolation='none')
        #plt.xlabel('subband')
        #plt.ylabel('time')
        #plt.colorbar()
        #plt.savefig('xy.png')

        return XX_noise,XY_noise,YX_noise,YY_noise,XX,XY,YX,YY



#sim=LOFARvis(n_time=1024,n_freq=512,n_rfi=10)
#sim.generate_data()
