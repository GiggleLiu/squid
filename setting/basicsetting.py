from numpy import *
from setting.basicsetting import *
#parameter setting
#natural units -
#Energy: GHz
#Charge: e
#Flux: hbar/e (hbar=1)

#only SQUID_... will be passed to the program
aQ=0.63
bQ=0.15
kQ=0.       #(Ic1-Ic2)/(Ic1+Ic2)
sQ=0.       #2Cs/(C1+C2)
Ej=50.      #(Ic1+Ic2)/4
Ec=1.       #1/(C1+C2)

SQUID_IC1=2.*(1+kQ)*Ej
SQUID_IC2=2.*(1-kQ)*Ej
SQUID_IC3=aQ*(SQUID_IC1+SQUID_IC2)/2

SQUID_C1=SQUID_IC1*Ec/Ej/4.
SQUID_C2=SQUID_IC2*Ec/Ej/4.
SQUID_C3=SQUID_IC3*Ec/Ej/4.
SQUID_CS=(SQUID_C1+SQUID_C2)*sQ/2
SQUID_L=bQ/2*(1/SQUID_IC1+1/SQUID_IC2+1/SQUID_IC3)

#Model setting
MODEL_NA=5
MODEL_NS=10
MODEL_NT=2

#plot setting
NPHI=101
PHIMIN=pi*0.
PHIMAX=pi*1.
LOWEST_NBAND=14

#global setting
MODEL_FOLDER='TriJJ'
APPENDEK=True
DEBUG=False
