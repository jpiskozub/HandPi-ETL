import random

import numpy as np
import pandas as pd
import tsaug
import pendulum
import psycopg as psql
from tsaug.visualization import plot

import configparser
from scipy import signal
import sklearn.preprocessing
config = configparser.ConfigParser()



ADC_channels = ['P1_1', 'P1_2', 'P2_1', 'P2_2', 'P3_1', 'P3_2', 'P4_1', 'P4_2', 'P5_1', 'P5_2']
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

config.read('config.ini')
SAMPLE_SIZE = 75
def calculate_timedelta(gesture_df):
    gesture_tmstmp = gesture_df.iloc[:, 18]
    gesture_tmstmp_diff = np.diff(gesture_tmstmp)

    gesture_tmstmp_period = [np.timedelta64(0)]

    for i in range(gesture_tmstmp_diff.size):
        if i==0:
            gesture_tmstmp_period.append(gesture_tmstmp_diff[i] + np.timedelta64(gesture_tmstmp_period[i - 1]))
        else:
            gesture_tmstmp_period.append((gesture_tmstmp_diff[i] + gesture_tmstmp_diff[i - 1])+ np.timedelta64(gesture_tmstmp_period[i - 1]))
    gesture_df.iloc[:, 18] =  gesture_tmstmp_period
    return gesture_df
#%%    
def get_rand_gesture(sample_size, show_results = True):
    with psql.connect(dbname=config['DB']['dbname'], user=config['DB']['user'], password=config['DB']['password'], host=config['DB']['dbpi_ip_addr']) as psqlconn:
        psqlcur = psqlconn.cursor()

        psqlcur.execute('SELECT count(*) FROM static_gestures;')
        max_size_stat_gest = psqlcur.fetchone()
        psqlcur.execute('SELECT count(*) FROM dynamic_gestures;')
        max_size_dyn_gest = psqlcur.fetchone()


        rand_stat_gest = random.randrange(1, max_size_stat_gest[0], sample_size)
        rand_dyn_gest = random.randrange(1, max_size_dyn_gest[0], sample_size)

        gesture_type = random.choice(['static_gestures', 'dynamic_gestures'])
        if gesture_type == 'static_gesture':
            psqlcur.execute('SELECT * FROM {} OFFSET (%s) LIMIT (%s);'.format(gesture_type), (rand_stat_gest, sample_size,))
        else:
            psqlcur.execute('SELECT * FROM {} OFFSET (%s) LIMIT (%s);'.format(gesture_type), (rand_dyn_gest, sample_size,))
        gesture = pd.DataFrame(psqlcur.fetchall())
    gesture = calculate_timedelta(gesture)
    gesture.columns = ['exam_id', *ADC_channels, *IMU_channels, 'sign', 'timestamp', 'gesture_id']
        
    if show_results == True:
        gesture.plot(x = 'timestamp', y = [*IMU_channels,*ADC_channels], subplots =[ADC_channels, IMU_channels[0:3], IMU_channels[3:6]])
        
    return gesture


def augment_gesture(gesture_df, nr_of_reps):
    gesture_buf_full = gesture_df.to_numpy()
    gesture_buf = gesture_buf_full[:, 1:17]
    gesture_buf = gesture_buf.astype(float)

    gesture_buf_3d = gesture_buf.reshape(1, gesture_df.shape[0], 16)
    gesture_buf_3d_mask = gesture_buf.reshape(1, gesture_df.shape[0], 16)

    plot(gesture_buf_3d)
    
    augmenter_values = {
                        'n_speed_change':2,
                        'max_speed_ratio':2,
                        'max_drift':1,
                        'n_drift_points':2,
                        'noise_scale':0.05
                        }


    my_augmenter = (
            tsaug.TimeWarp(n_speed_change=augmenter_values['n_speed_change'], max_speed_ratio=augmenter_values['max_speed_ratio']) * nr_of_reps  # random time warping 5 times in parallel
            # tsaug.Quantize(n_levels=[1000, 5000, 15000])  # random quantize to 10-, 20-, or 30- level sets
            + tsaug.Drift(max_drift=augmenter_values['max_drift'], n_drift_points=augmenter_values['n_drift_points'], kind='additive', per_channel=True, normalize=True, seed=None) @ 0.8  # with 80% probability, random drift the signal up to 10% - 50%
            # tsaug.Convolve(window='hann', size=7, per_channel=False, repeats=1, prob=1.0, seed=None)
            # tsaug.Reverse() @ 0.5  # with 50% probability, reverse the sequence
            # tsaug.Crop(70, resize=100, repeats=1, prob=1.0)
            + tsaug.AddNoise(scale=augmenter_values['noise_scale'])
                    )



    gesture_augmented = my_augmenter.augment(gesture_buf_3d, gesture_buf_3d_mask)

    gesture_augmented_buf = gesture_augmented[0].reshape(nr_of_reps, gesture_df.shape[0], 16)
    plot(gesture_augmented_buf)
    
    gesture_augmented_df = pd.DataFrame(np.column_stack((gesture_augmented_buf[1,:,:],gesture_df.iloc[:,17:20])))
    
    return gesture_augmented_df





def upload_augmented_gesture(gesture_augmented_df):
    gesture_augmented_df = pd.DataFrame(gesture_augmented_df)
    with psql.connect(dbname=config['DB']['dbname'], user=config['DB']['user'], password=config['DB']['password'], host=config['DB']['dbpi_ip_addr']) as psqlconn:
        psqlcur = psqlconn.cursor()
        SQL_augmented_insert = 'INSERT INTO augmented_gestures (p1_1, p1_2, p2_1, p2_2, p3_1, p3_2, p4_1, p4_2, p5_1, p5_2, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z, gesture, tmstmp_interval, aug_id, parent_id ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'

        psqlcur.execute("SELECT MAX(aug_id) FROM augmented_gestures;") #May be done with RETURN SQL statement - To be explored
        last_id=psqlcur.fetchone()

        if last_id[0] == None:
            id_column = [1 for i in range(gesture_augmented_df.shape[0])]
        else:
            id_column = [last_id[0]+1 for i in range(gesture_augmented_df.shape[0])]

        gesture_augmented_df.insert(19,'aug_id',id_column, True)

        for i in range(gesture_augmented_df.shape[0]):
            psqlcur.execute(SQL_augmented_insert,gesture_augmented_df.values[i,:].tolist())
 # %%           
def detect_shortcircuit(gesture_df, shtct_threshold):
    if True in list(gesture_df.iloc[:,1:11].ge(shtct_threshold).any()):
        return True
    else:
        return False
# %%
def ADC_smoothing(gesture_df, show_results=True):
    gesture_df.plot(x = 'timestamp', y = [*ADC_channels])
    
    WINDOW_SIZE = 7
    
    for (colname, colvals) in gesture_df.iloc[:,1:11].items():
        
        gesture_df.loc[0:64,colname] = pd.Series(signal.medfilt(colvals.values,WINDOW_SIZE))
        
    #for (colname, colvals) in gesture_df.iloc[65:,0:9].items():
        #gesture_df[colname][65:] = signal.medfilt(colvals.values,WINDOW_SIZE)
        
    gesture_df.plot(x = 'timestamp', y = [*ADC_channels])

# %%
def IMU_smoothing(gesture_df, show_results=True):
    if show_results == True:
        gesture_df.plot(x = 'timestamp', y = [*IMU_channels], subplots =[IMU_channels[0:3], IMU_channels[3:6]])
        
    ACC_WINDOW_SIZE = 31
    GYRO_WINDOW_SIZE = 7
    
    for (colname, colvals) in gesture_df.iloc[:,9:13].items():
        gesture_df[colname] = signal.medfilt(colvals.values,ACC_WINDOW_SIZE)
        
    for (colname, colvals) in gesture_df.iloc[:,13:17].items():
        gesture_df[colname] = signal.medfilt(colvals.values,GYRO_WINDOW_SIZE)
        
    if show_results == True:    
        gesture_df.plot(x = 'timestamp', y = [*IMU_channels], subplots =[IMU_channels[0:3], IMU_channels[3:6]])
# %%
def normalize_data(gesture_df, show_results=True):
    if show_results == True:
        gesture_df.plot(x = 'timestamp', y = [*IMU_channels,*ADC_channels], subplots =[ADC_channels, IMU_channels[0:3], IMU_channels[3:6]])
        
    #normalizer = 
    #min_max_scaler = preprocessing.MinMaxScaler()
    #for (colname, colvals) in gesture_df.iloc[:,0:17].items():
    #    gesture_df[colname] = min_max_scaler.fit_transform(colvals.values.reshape(-1, 1))
    for (colname, colvals) in gesture_df.iloc[:,1:11].items():
        gesture_df[colname] = preprocessing.normalize(colvals.values.reshape(-1, 1), norm = 'max')
    
        
    if show_results == True:    
        gesture_df.plot(x = 'timestamp', y = [*IMU_channels,*ADC_channels], subplots =[ADC_channels, IMU_channels[0:3], IMU_channels[3:6]])
    
# %%
def get_exam_gestures(sample_size):
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    

    with psql.connect(dbname=config['DB']['dbname'], user=config['DB']['user'], password=config['DB']['password'], host=config['DB']['dbpi_ip_addr']) as psqlconn:
        psqlcur = psqlconn.cursor()
        
        
        psqlcur.execute('SELECT * FROM static_gestures;')
        static_gestures = pd.DataFrame(psqlcur.fetchall())
        psqlcur.execute('SELECT * FROM dynamic_gestures;')
        dynamic_gestures = pd.DataFrame(psqlcur.fetchall())
    gestbase = pd.concat([static_gestures,dynamic_gestures], ignore_index = True)
    gestbase.columns = ['exam_id', *ADC_channels, *IMU_channels, 'sign', 'timestamp', 'gesture_id']
    return gestbase
# %%
def save2csv(gesture_df,sample_size):



    gest_csv = pd.DataFrame()
    gest_csv_l = []
    gest_shtct_csv = pd.DataFrame()
    gest_shtct_csv_l = []
    for i in range(0,gesture_df.shape[0],sample_size):
        gest = gesture_df.iloc[i:i+sample_size]
        if not detect_shortcircuit(gest, 24000):
            gest_csv_l.append(gest)
        else:
            gest_shtct_csv_l.append(gest)
    gest_csv = pd.concat(gest_csv_l, ignore_index = True)
    gest_shtct_csv = pd.concat(gest_shtct_csv_l, ignore_index = True)


    return gest_csv.to_csv('gesty.csv', index=False),gest_shtct_csv.to_csv('gesty_zwarcia.csv', index=False)



# %%
#def get_all_gestures