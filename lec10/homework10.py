import numpy as np
import torch, torch.nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    
    '''
    L = int(0.004*Fs)
    S = int(0.002*Fs)
    mstft = np.abs(np.fft.fft(np.array([waveform[m+1:m+1+L]-waveform[m:m+L] for m in range(0, len(waveform)-L, S)]), axis=1))
    features = 20*np.log10(np.maximum(0.001*np.amax(mstft), mstft))[:, 0:int(L/2)]

    framelength = int(0.025*Fs)
    frameskip = int(0.01*Fs)
    energy = np.sum(np.square(np.array([waveform[m:m+framelength] for m in range(0, len(waveform)-framelength, frameskip)])), axis=1)
    vad = [1 if energy[m] > 0.1*max(energy) else 0 for m in range(len(energy))]
    startframes = [m for m in range(1, len(vad)) if vad[m-1]==0 and vad[m]==1]
    endframes = [m for m in range(1, len(vad)) if vad[m-1]==1 and vad[m]==0]

    labels = np.zeros(len(features))
    for num in range(1, len(startframes)+1):
        labels[5*startframes[num-1]:5*endframes[num-1]+4] = num

    return features, labels

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step.
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    NFEATS = features.shape[1]
    NLABELS = 1 + int(max(labels))
    model = torch.nn.Sequential(
        torch.nn.LayerNorm(NFEATS),
        torch.nn.Linear(NFEATS, NLABELS)
    ).double()
    lossfunction = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    lossvalues = np.zeros(iterations)
    for t in range(iterations):
        z = model(torch.tensor(features, dtype=torch.float64))
        loss = lossfunction(z, torch.tensor(labels, dtype=torch.long))
        lossvalues[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, lossvalues

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    z = model(torch.tensor(features, dtype=torch.float64))
    probabilities = z.softmax(dim=-1).detach().numpy()
    return probabilities

