import numpy as np


def percent_change(x):
    z = np.zeros(x.shape[:])
    z[0]=0
    for t in range(len(x)):
        if t>=1:
            z[t]=(x[t]-x[t-1])/(np.abs(x[t-1])+1e-6)
    return z[:]

def trend_decomp(sig,N):
    part = int(len(sig)/N)
    i=0
    j=0
    slopes = np.zeros(N+1)
    intercepts = np.zeros(N+1)
    while i<len(sig):
        end_i = i+part
        if end_i>len(sig):
            end_i = len(sig)
        sig_part = sig[i:end_i]
        x = np.column_stack((np.ones((len(sig_part),1)),np.arange(0,len(sig_part))))
        A=x.T.dot(x)
        b=x.T.dot(sig_part)
        z = np.linalg.solve(A,b)
        slopes[j] = z[1]
        intercepts[j] = z[0]
        i=i+part
        j+=1

    return intercepts,slopes

def background_noise(sig,W):
    sig = sig[:-4]-np.convolve(sig,np.ones(5)/5,mode='valid')
    sig = sig-np.min(sig)+1e-10
    N = len(sig)
    f = np.zeros(W)
    for n in range(0,5000):
        i = np.random.randint(0,N-W)
        f = f*(n)/(n+1) + np.fft.fftshift(np.fft.fft(sig[i:i+W])*1/(n+1))
    f = np.absolute(np.fft.ifft(f))
    limiter = np.max(f)
    f = np.log(f+limiter*0.01) - np.log(limiter*0.01)
    return f
	
def get_sse_sst(y,y_pred):
    sse = np.sum((y.flatten()-y_pred.flatten())**2)
    sst = np.sum((y.flatten()-np.mean(y))**2)
    return sse,sst

def get_mae(y,y_pred):
    mae = np.mean(np.abs(y.flatten()-y_pred.flatten()))
    return mae

def get_mape(y,y_pred):
    mape = 100*np.abs(y.flatten()-y_pred.flatten())/np.abs(y.flatten())
    mape = np.mean(mape)
    return mape

def get_smape(y,y_pred):
    mape = 200*np.abs(y.flatten()-y_pred.flatten())/np.abs(y.flatten()+y_pred.flatten())
    mape = np.mean(mape)
    return mape