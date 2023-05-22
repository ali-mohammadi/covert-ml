import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

chl = 'rician'

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def generate_signals(n):
  signals = np.random.rand(n, 2)
  labels = np.zeros((n,))
  if mod == 'bpsk':
    signals[signals[:, 0] < 0.5, 0] = -1
    signals[signals[:, 0] >= 0.5, 0] = 1
    signals[:, 1] = 0
  if mod == 'qpsk':
    for i in range(2):
      signals[signals[:, i] < 0.5, i] = -0.7
      signals[signals[:, i] >= 0.5, i] = 0.7
  messages = np.unique(signals, axis=0)
  for key, value in enumerate(messages):
    idx = np.where(np.all(signals == value, axis=1))
    labels[idx] = key

  return signals, signals

def channel(x, snr):
  ebno = 10.0 ** (snr / 10.0)
  if chl == 'awgn':
    noise = np.random.randn(*x.shape) / ((2 * r * ebno) ** 0.5)
    return x + noise
  if chl == 'rayleigh' or chl == 'rician':
    noise = np.random.randn(*x.shape) / ((2 * r * ebno) ** 0.5)
    if chl == 'rayleigh':
        h = np.random.normal(0, np.sqrt(1/2), size=(x.shape[0], 2)).view(complex)
    else:
        k = 1
        std = (1 / (k + 1)) ** 0.5
        mean = (k / 2 * (k + 1)) ** 0.5
        h = np.random.normal(mean, std, size=(x.shape[0], 2)).view(complex)
    x = x.view(complex)
    y = h * x
    return ((y.view(float) + noise).view(complex) / h).view(float)

def decode(x):
  if mod == 'bpsk':
    x[x[:, 0] < 0, 0] = -1
    x[x[:, 0] >= 0, 0] = 1
    x[:, 1] = 0
  if mod == 'qpsk':
    for i in range(2):
      x[x[:, i] < 0, i] = -0.7
      x[x[:, i] >= 0, i] = 0.7
  return x

def ber(x, y):
  return np.sum(np.any(x != y, axis=1)) / len(y)


bers = [[], []]
for key, _mod in enumerate(['bpsk', 'qpsk']):
  mod = _mod
  r = 1 if mod == 'bpsk' else 2
  x, y = generate_signals(50 * 1024)
  if chl == 'awgn':
    snr_range = list(frange(-4, 9, 1.5))
  else:
    snr_range = list(frange(0, 22, 5))
  for snr in snr_range:
    bers[key].append(ber(decode(channel(x, snr)), y))

print(bers)

for ber in bers:
  plt.plot(snr_range, ber)

plt.yscale('log')
plt.show()