import subprocess
import os

system = 'mimo'
current_dir = os.getcwd()
# for t in range(0, 11):
#     os.chdir(current_dir + '/wireless_autoencoder' + '/' + system)
#     p = subprocess.Popen('python train.py ' + str(t), shell=True).wait()
os.chdir(current_dir + '/wireless_covert' + '/' + system)
for i in range(1, 2):
    for j in range(2, 5):
        for k in range(1, 9):
           p = subprocess.Popen('python train2.py ' + str(0) + ' ' + ' '.join(str(x / 10) for x in [i, j, k]), shell=True).wait()
    # p = subprocess.Popen('python train2.py ' + str(t) + ' ' + ' '.join(str(x / 10) for x in [1, 8, 1]), shell=True).wait()
