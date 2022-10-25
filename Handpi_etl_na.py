import random

import numpy as np
import pandas as pd
import tsaug
import pendulum
import psycopg as psql
from tsaug.visualization import plot


#%%
dbpi_ip_addr = '192.168.0.100'

ADC_channels=['P1_1', 'P1_2', 'P2_1', 'P2_2', 'P3_1', 'P3_2', 'P4_1', 'P4_2', 'P5_1', 'P5_2']
IMU_channels = ['Euler_x', 'Euler_y', 'Euler_z', 'Acc_x', 'Acc_y', 'Acc_z']

sign_types = ['static', 'dynamic']
sign_types_dict = {'a': sign_types[0],
                   'ą': sign_types[1],
                   'b': sign_types[0],
                   'c': sign_types[0],
                   'ć': sign_types[1],
                   'ch': sign_types[1],
                   'cz': sign_types[1],
                   'd': sign_types[1],
                   'e': sign_types[0],
                   'ę': sign_types[1],
                   'f': sign_types[1],
                   'g': sign_types[1],
                   'h': sign_types[1],
                   'i': sign_types[0],
                   'j': sign_types[1],
                   'k': sign_types[1],
                   'l': sign_types[0],
                   'ł': sign_types[1],
                   'm': sign_types[0],
                   'n': sign_types[0],
                   'ń': sign_types[1],
                   'o': sign_types[0],
                   'ó': sign_types[1],
                   'p': sign_types[0],
                   'r': sign_types[0],
                   'rz': sign_types[1],
                   's': sign_types[0],
                   'ś': sign_types[1],
                   'sz': sign_types[1],
                   't': sign_types[0],
                   'u': sign_types[0],
                   'w': sign_types[0],
                   'y': sign_types[0],
                   'z': sign_types[1],
                   'ź': sign_types[1],
                   'ż': sign_types[1]}

#%%

with psql.connect(dbname = 'handpi', user = 'handpi', password = 'raspberryhandpi', host = dbpi_ip_addr) as psqlconn:
    psqlcur = psqlconn.cursor()

    psqlcur.execute('SELECT count(*) FROM static_gestures;')
    max_size_stat_gest = psqlcur.fetchone()
    psqlcur.execute('SELECT count(*) FROM dynamic_gestures;')
    max_size_dyn_gest = psqlcur.fetchone()

    sample_size = 100
    rand_stat_gest = random.randrange(1, max_size_stat_gest[0], sample_size)
    rand_dyn_gest = random.randrange(1, max_size_dyn_gest[0], sample_size)

    gesture_type=random.choice(['static_gestures', 'dynamic_gestures'])
    if gesture_type == 'static_gesture':
        psqlcur.execute('SELECT * FROM {} OFFSET (%s) LIMIT (%s);'.format(gesture_type), (rand_stat_gest, sample_size,))
    else:
        psqlcur.execute('SELECT * FROM {} OFFSET (%s) LIMIT (%s);'.format(gesture_type), (rand_dyn_gest, sample_size,))


#%%
gesture = pd.DataFrame(psqlcur.fetchall())

#%%
my_augmenter = (
                tsaug.TimeWarp(n_speed_change=1, max_speed_ratio=2)*5 # random time warping 5 times in parallel
                #tsaug.Quantize(n_levels=[100, 500, 1500])  # random quantize to 10-, 20-, or 30- level sets
                #tsaug.Drift(max_drift=0.01, n_drift_points=20, kind='additive', per_channel=True, normalize=True, repeats=2, prob=1.0, seed=None) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
                #tsaug.Convolve(window='hann', size=7, per_channel=False, repeats=1, prob=1.0, seed=None)
                #tsaug.Reverse() @ 0.5  # with 50% probability, reverse the sequence
                #tsaug.Crop(70, resize=100, repeats=1, prob=1.0)
                #+tsaug.AddNoise(scale=0.005)
                )
#%%
gesture_buf = gesture.to_numpy()
gesture_buf = gesture_buf[:,1:17]
gesture_buf=gesture_buf.astype(float)
#%%

np.savetxt('test_buf.csv',gesture_buf)
#%%
gesture_buf_3d = gesture_buf.reshape(1,sample_size,16)
gesture_buf_3d_mask = gesture_buf.reshape(1,sample_size,16)

plot(gesture_buf_3d)
#%%
gesture_augmented = my_augmenter.augment(gesture_buf_3d,gesture_buf_3d_mask)
#%%
gesture_augmented_buf=gesture_augmented[0].reshape(5,sample_size,16)
plot(gesture_augmented_buf)

#np.savetxt('test.csv',gesture_augmented)