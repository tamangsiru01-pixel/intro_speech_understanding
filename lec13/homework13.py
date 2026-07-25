import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    frames = np.array([speech[m*frame_skip:m*frame_skip+frame_length] for m in range(int((len(speech)-frame_length)/frame_skip))])
    A = librosa.lpc(frames, order=order)
    nframes, nsamps = frames.shape
    excitation = np.zeros((nframes, nsamps))
    for frame in range(nframes):
        for samp in range(order, nsamps):
            for k in range(order+1):
                excitation[frame, samp] += A[frame, k] * frames[frame, samp-k]
    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    order = A.shape[1] - 1
    synthesis = np.zeros(len(e))
    for n in range(len(e)):
        frame = int(n/frame_skip)
        synthesis[n] = e[n]
        for k in range(1, min(n, order+1)):
            synthesis[n] -= A[frame, k] * synthesis[n-k]
    return synthesis

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    gain = np.sqrt(np.average(np.square(excitation), axis=1))
    nframes = len(excitation)
    total_len = nframes * frame_skip
    p = np.zeros(total_len)
    p[::T0] = 1
    e_robot = np.zeros(total_len)
    for n in range(total_len):
        frame = int(n/frame_skip)
        e_robot[n] = gain[frame] * p[n]
    return gain, e_robot

