import pyaudio
import numpy as np
import pylab
import time
import scipy.signal as signal
import iirFIlter as iir
RATE = 44100
CHUNK = int(RATE/20) # RATE / number of updates per second

fullUnfilteredData = []
fullFilteredData = []



def soundplot(stream):
    global fullFilteredData
    global fullUnfilteredData
    t1=time.time()
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    ##STart filtering
    ## Take FFT to show signal in the
    ##Create a clean array to store the filtered data
    clean_data = np.zeros(len(data))
    clean_data = data
    # Creation of the coeffs for high-pass
    rate = 44100
    fc= 2000
    cut_off = 20
    fc = fc/rate
    f1 = cut_off / rate * 2
    sos = signal.butter(6, [f1], 'high', output='sos')
    IIR = iir.IIRFilter(sos)
    filtered_data = abs(IIR.dofilter(data))
    ##Plot Unfiltered Data
    pylab.plot(data)
    pylab.title(i)
    pylab.grid()
    pylab.axis([0,len(data),-2**16/2,2**16/2])
    pylab.savefig("03.png",dpi=50)
    ##Plot filtered Data
    pylab.plot(filtered_data)
    pylab.title(i)
    pylab.grid()
    pylab.axis([0, len(filtered_data), -2 ** 16 / 2, 2 ** 16 / 2])
    pylab.savefig("04.png", dpi=50)

    fullFilteredData.append(filtered_data)
    fullUnfilteredData.append(data)


    pylab.close('all')
    print("took %.02f ms"%((time.time()-t1)*1000))



if __name__=="__main__":
    p=pyaudio.PyAudio()
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNK)
    for i in range(int(10*(2**13)/1024)): #do this for a few seconds, can't get precise running times as it seems to be depending on PC performance
        soundplot(stream)
        data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
        data = data * np.hanning(len(data))  # smooth the FFT by windowing data
        fft = abs(np.fft.fft(data).real)
        fft = fft[:int(len(fft) / 2)]  # keep only first half
        freq = np.fft.fftfreq(CHUNK, 1.0 / RATE)
        freq = freq[:int(len(freq) / 2)]  # keep only first half
        freqPeak = freq[np.where(fft == np.max(fft))[0][0]] + 20
        print("peak frequency: %d Hz" % freqPeak)


    pylab.plt.plot(freq, fft)
    pylab.plt.axis([0,4000,None,None])
    pylab.plt.show()
    pylab.plt.close()
        #time.sleep(1)
    stream.stop_stream()
    stream.close()
    p.terminate()

    ##Peak Detection
    # for i in range(10):  # to it a few times just to see
    #     data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    #     data = data * np.hanning(len(data))  # smooth the FFT by windowing data
    #     fft = abs(np.fft.fft(data).real)
    #     fft = fft[:int(len(fft) / 2)]  # keep only first half
    #     freq = np.fft.fftfreq(CHUNK, 1.0 / RATE)
    #     freq = freq[:int(len(freq) / 2)]  # keep only first half
    #     freqPeak = freq[np.where(fft == np.max(fft))[0][0]] + 1
    #     print("peak frequency: %d Hz" % freqPeak)

        #uncomment this if you want to see what the freq vs FFT looks like
        # plt.plot(freq,fft)
        # plt.axis([0,4000,None,None])
        # plt.show()
        # plt.close()

    ##Plot Full sampled data
    pylab.plot(fullUnfilteredData)
    pylab.title("Full time series of the sampled data")
    pylab.grid()
    pylab.axis([0, len(fullUnfilteredData), -2 ** 16 / 2, 2 ** 16 / 2])
    pylab.xlabel("Time")
    pylab.ylabel("Amplitude")
    pylab.show()
    ##Plot Full sampled data after filtering
    pylab.plot(fullFilteredData)
    pylab.title("Full time series of the filtered data")
    pylab.grid()
    pylab.axis([0, len(fullFilteredData), -2 ** 16 / 2, 2 ** 16 / 2])
    pylab.xlabel("Time")
    pylab.ylabel("Amplitude")
    pylab.savefig("06.png", dpi=50)
    pylab.show()

    # ##Plot Unfiltered Data in the freq domain
    # fftUnfiltered = abs(np.fft.fft(fullFilteredData))
    # fftUnfiltered = fftUnfiltered.real
    # ##Remove mirror
    # fftUnfiltered = fftUnfiltered[:int(len(fftUnfiltered) / 2)]
    # pylab.plot(fftUnfiltered)
    # pylab.title("Filtered Time series in the freq domain")
    # pylab.grid()
    # pylab.axis([0, len(fftUnfiltered), -2 ** 16 / 2, 2 ** 16 / 2])
    # pylab.xlabel("Frequency")
    # pylab.ylabel("Ampplitude")
    # pylab.savefig("07.png", dpi=50)
    # pylab.show()
    #
    #
    # ##Plot Filtereddata in the freq domain
    # fftFiltered = abs(np.fft.fft(fullFilteredData))
    # fftFiltered = fftFiltered.real
    # ##Remove mirror
    # fftFiltered = fftFiltered[:int(len(fftFiltered) / 2)]
    # pylab.plot(fftFiltered)
    # pylab.title("Filtered Time series in the freq domain")
    # pylab.grid()
    # pylab.axis([0, len(fftFiltered), -2 ** 16 / 2, 2 ** 16 / 2])
    # pylab.savefig("08.png",dpi=50)
    # pylab.show()
