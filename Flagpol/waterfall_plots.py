#! /usr/bin/env python

import argparse
import numpy as np
import casacore.tables
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

fig=plt.figure(1)
fig.tight_layout()

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

def show_ms_waterfall(msname,datacol,apply_flags,window_size,plot_type='P',save_fig=False,baseline=None,range_fraction=1.0):
    if baseline:
        ant1=baseline[0][0]
        ant2=baseline[0][1]

    t=casacore.tables.table(msname,readonly=True)
    for t1 in t.iter(['ANTENNA1','ANTENNA2']):
        a1=t1.getcol('ANTENNA1')[0]
        a2=t1.getcol('ANTENNA2')[0]
        if baseline:
           if (a1 != ant1) or (a2 != ant2):
              continue

        tt=t1.sort(sortlist='TIME')
        vl=tt.getcol(datacol)
        if not apply_flags:
            xx=vl[:,:,0]
            xy=vl[:,:,1]
            yx=vl[:,:,2]
            yy=vl[:,:,3]
        else:
            fl=tt.getcol('FLAG')
            xx=vl[:,:,0]*(1-fl[:,:,0])
            xy=vl[:,:,1]*(1-fl[:,:,1])
            yx=vl[:,:,2]*(1-fl[:,:,2])
            yy=vl[:,:,3]*(1-fl[:,:,3])

        ii=np.real(xx+yy)
        qq=np.real(xx-yy)
        uu=np.real(xy+yx)
        vv=np.real(1j*(xy-yx))
        W=window_size
        if W>1: # only divide if window size > 1, otherwise all=1
           pp=np.sqrt((qq)**2+(uu)**2+(vv)**2)
           qq=qq/pp
           uu=uu/pp
           vv=vv/pp
        D=qq.shape[1]//W
        T=qq.shape[0]
        if plot_type=='I':
           ii1=np.zeros((T,D))
           for ct in range(T):
             for ci in range(D):
               ii1[ct,ci]=np.sum(ii[ct,W*ci:W*(ci+1)])
           pp=np.log(np.abs(ii1))
        else: # plot_type=='P'
           qq1=np.zeros((T,D))
           uu1=np.zeros((T,D))
           vv1=np.zeros((T,D))
           for ct in range(T):
             for ci in range(D):
               qq1[ct,ci]=np.sum(qq[ct,W*ci:W*(ci+1)])
               uu1[ct,ci]=np.sum(uu[ct,W*ci:W*(ci+1)])
               vv1[ct,ci]=np.sum(vv[ct,W*ci:W*(ci+1)])

           pp=np.sqrt(qq1**2+uu1**2+vv1**2)
    
        plt.imshow(pp,aspect='auto')
        plt.xlabel('Channel',fontdict=font)
        plt.ylabel('Time sample',fontdict=font)
        cbar=plt.colorbar()
        plt.clim(range_fraction*pp.min(),range_fraction*pp.max())
        if plot_type=='I':
           cbar.set_label('Log intensity',fontdict=font)
        else:
           cbar.set_label('Polarization',fontdict=font)
        plt.suptitle('ANT '+str(a1)+' ANT '+str(a2))
        plt.pause(0.5)
        if save_fig:
            plt.savefig('ANT'+str(a1)+'_ANT'+str(a2)+'.png')
        plt.clf()


def ant_pair(arg):
    # For simplity, assume arg is a pair of integers
    # separated by a comma.
    return [int(x) for x in arg.split(',')]

if __name__=='__main__':
    parser=argparse.ArgumentParser(
      description='Show visibilities as waterfall plots',
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--MS',type=str,metavar='\'*.MS\'',
        help='MS to use')
    parser.add_argument('--col',type=str,default='DATA',metavar='d',
        help='Column name DATA|MODEL_DATA|...')
    parser.add_argument('--plot_type',type=str,default='P',metavar='p',
        help='Plot type I (log Stokes amp) P (Polarization)')
 
    parser.add_argument('--apply_flags', default=True, action=argparse.BooleanOptionalAction, help='Apply flags to data before showing')
    parser.add_argument('--window_size',type=int,default=2,metavar='w',
        help='Window size (in channels) to average')
    parser.add_argument('--range_fraction',type=float,default=1,metavar='r',
        help='Limite colormap to this fraction*[min,max]')
    parser.add_argument('--save', default=False, action=argparse.BooleanOptionalAction, help='Save plot to file')
    parser.add_argument('--baseline', default=None, type=ant_pair, nargs='+', help='Only plot the baseline for given antenna pair, e.g. --baseline 0,11')

    args=parser.parse_args()

    if args.MS:
        show_ms_waterfall(args.MS,args.col,args.apply_flags,args.window_size,plot_type=args.plot_type,baseline=args.baseline,save_fig=args.save,range_fraction=args.range_fraction)
    else:
        parser.print_help()
