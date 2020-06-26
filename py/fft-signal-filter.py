import numpy as np 
import matplotlib.pyplot as plot
plot.rcParams['figure.figsize'] = [10,5]
plot.rcParams.update({'font.size':12})

dt = 0.001 #sampling every 1/1000 second
t = np.arange(0,1,dt)
f_clean = np.sin(2*np.pi*8*t) #insert your formula here. current formula = sin(2pi*8t) [8 = omega]
f = f_clean + 3*np.random.randn(len(t)) # adding random noise upon the function

n = len(t)
fhat = np.fft.fft(f,n) # performing fast fourier transform
PSD = fhat * np.conj(fhat) / n #determining the power spectrum from the function
freq = (1/dt*n) * np.arange(n) 
L = np.arange(1,np.floor(n/2), dtype ='int') # creating a single side fft
indices = PSD > 100 # filtering the high spectrum
PSDfix = PSD * indices #applying the filter
fhat_new = indices * fhat # applying the filter to the fft function
ffilt = np.fft.ifft(fhat_new) # inverse fft

fig, axs = plot.subplots(3,1) #plotting 3 graphs

plot.sca(axs[0]) #plot for first graph
plot.plot(t, f, color = 'c', LineWidth = 1.5, label = 'with noise') #plotting the f+noise
plot.plot(t, f_clean, color = 'g', LineWidth = 2.5, label = 'original') #plotting the clean original function
plot.xlim(t[0], t[-1])
plot.ylim(-7.5, 7.5)
plot.xlabel ('time')
plot.ylabel('amplitude')
plot.legend()

plot.sca(axs[1]) #plot for the second graph
plot.plot(freq[L], PSD[L], color = 'c', LineWidth = 1.5, label = 'with noise') #plotting the Power Spectral Density of the function with noise
plot.plot(freq[L], PSDfix[L], color = 'k', LineWidth = 1.5, label = 'filtered') # plotting the filtered function's PSD
plot.xlim(freq[L[0]], freq[L[-1]])
plot.xlabel ('freq')
plot.ylabel('power')
plot.legend()

plot.sca(axs[2]) #plot for the thir graph
plot.plot(t, f, color = 'c', LineWidth = 1.5, label = 'with noise') #plotting the unfiltered function
plot.plot(t, ffilt, color = 'k', LineWidth = 1.5, label = 'filtered') #plotting the filtered function
plot.xlim(t[0], t[-1])
plot.xlabel ('time')
plot.ylabel('amplitude')
plot.legend()

plot.subplots_adjust(hspace=0.5)
plot.show() #actually plotting the three graph into a window.
