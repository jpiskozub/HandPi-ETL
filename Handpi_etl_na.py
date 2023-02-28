#%%
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
from ast import literal_eval
from functools import wraps


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
#%%
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


def  augment_gesture(gesture_df, nr_of_reps):
    gesture_df.fillna(method='backfill', inplace=True)

    gesture_buf_full = gesture_df.to_numpy()

    gesture_buf = (gesture_buf_full[:, 1:17])
    gesture_buf = gesture_buf.astype(float)

    gesture_buf_3d = gesture_buf.reshape(1, gesture_df.shape[0], 16)
    gesture_buf_3d_mask = gesture_buf.reshape(1, gesture_df.shape[0], 16)

    #plot(gesture_buf_3d)
    
    augmenter_values = {
                        'n_speed_change':2,
                        'max_speed_ratio':2,
                        'max_drift':1,
                        'n_drift_points':2,
                        'noise_scale':0.03
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
    #plot(gesture_augmented_buf)

    sign_col = np.array(gesture_df['sign'])
    exam_id_col = np.array(gesture_df['exam_id'])
    tmstp_col = np.array(gesture_df['timestamp'])

    for i in range(nr_of_reps-1):
        sign_col = np.hstack((sign_col,gesture_df['sign']))
        exam_id_col = np.hstack((exam_id_col,gesture_df['exam_id']))
        tmstp_col = np.hstack((tmstp_col,gesture_df['timestamp']))

    gesture_augmented_df = pd.DataFrame(np.column_stack((exam_id_col,gesture_augmented_buf.reshape(gesture_augmented_buf.shape[0]*gesture_augmented_buf.shape[1],gesture_augmented_buf.shape[2]),sign_col,tmstp_col)))
    
    return gesture_augmented_df


def read_csv(csv_path):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
           dataset_df = pd.read_csv(csv_path, converters={"exam_id": literal_eval})
           kwargs['dataset_df'] = dataset_df
           func(*args, **kwargs)
           return wrapper
        return decorator

#%%
def augment_dataset(dataset_df,nr_of_reps):
    
  
    time_Series = [dataset_df.iloc[i:i + SAMPLE_SIZE].reset_index(drop=True) for i in range(0, dataset_df.shape[0] - SAMPLE_SIZE + 1, SAMPLE_SIZE)]
    #time_Series = pd.concat(time_Series, axis=1).T

    aug_df = pd.DataFrame()
    
    for i in range(len(time_Series)):
        aug_df = pd.concat([aug_df,augment_gesture(time_Series[i],nr_of_reps)], ignore_index=True)
    
    return aug_df


#%%



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

def detect_shortcircuit(gesture_df, shtct_threshold):
    if True in list(gesture_df.iloc[:,1:11].ge(shtct_threshold).any()):
        return True
    else:
        return False

def ADC_smoothing(gesture_df, show_results):
    gesture_df.plot(x = 'timestamp', y = [*ADC_channels])
    
    WINDOW_SIZE = 7
    
    for (colname, colvals) in gesture_df.iloc[:,1:11].items():
        
        gesture_df.loc[0:64,colname] = pd.Series(signal.medfilt(colvals.values,WINDOW_SIZE))
        
    #for (colname, colvals) in gesture_df.iloc[65:,0:9].items():
        #gesture_df[colname][65:] = signal.medfilt(colvals.values,WINDOW_SIZE)
    if show_results:
        gesture_df.plot(x = 'timestamp', y = [*ADC_channels])


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

def select_shtct(gesture_df,sample_size):



    gest_l = []
    gest_shtct_l = []
    for i in range(0,gesture_df.shape[0],sample_size):
        gest = gesture_df.iloc[i:i+sample_size]
        if not detect_shortcircuit(gest, 24000):
            gest_l.append(gest)
        else:
            gest_shtct_l.append(gest)
    gest = pd.concat(gest_l, ignore_index = True)
    gest_shtct = pd.concat(gest_shtct_l, ignore_index = True)

    return gest, gest_shtct
    #return gest_csv.to_csv('gesty.csv', index=False),gest_shtct_csv.to_csv('gesty_zwarcia.csv', index=False)

def examid_remap(gesture_df):

    examid_map = {
    1  :('wd',1),
    3  :('jp',2),
    4  :('mp',3),
    5  :('mp',3),
    7  :('ku',4),
    8  :('ku',4),
    9  :('dd',5),
    10 :('dd',5),
    11 :('sw',6),
    12 :('zp',7),
    13 :('zp',7),
    14 :('zp',7),
    15 :('sw',6),
    19 :('ku',8),
    20 :('ku',8),
    21 :('ku',8),
    22 :('pp',9),
    23 :('pp',9),
    24 :('pp',9),
    25 :('mo',10),
    26 :('sk',11),
    27 :('mu',12),
    28 :('mu',12),
    29 :('mk',13),
    30 :('mk',13),
    31 :('mk',13),
    32 :('mk',13),
    33 :('mk',13),
    34 :('mo',14),
    35 :('mo',14),
    36 :('tt',15),
    37 :('mp',16),
    39 :('jp',2),
    40 :('ku',4),
    41 :('mk',13),
    42 :('mk',13),
    43 :('mo',10),
    44 :('mo',14),
    45 :('mp',3),
    46 :('mp',3),
    47 :('mp',3),
    48 :('mp',16),
    49 :('mu',12),
    50 :('mu',12),
    51 :('pp',9),
    52 :('sk',11),
    53 :('wd',1),
    54 :('wd',1),
    55 :('zp',7),
    58 :('jP',17),
    59 :('jP',17),
    60 :('jP',17),
    62 :('mp',16)
            }



    gesture_df['exam_id'] = gesture_df['exam_id'].map(examid_map)

def generate_datasets(SAMPLE_SIZE):

    df = get_exam_gestures(SAMPLE_SIZE)
    df, df_st = select_shtct(df,SAMPLE_SIZE)
    ADC_smoothing(df,SAMPLE_SIZE)
    IMU_smoothing(df,SAMPLE_SIZE)
    examid_remap(df)
    calculate_timedelta(df)
    df_aug = augment_dataset(df,5)
    df.to_csv('gesty.csv', index=False)
    df_st.to_csv('gesty_zwarcia.csv', index=False)
    df_aug.to_csv('gesty_aug.csv', index=False)




# # %%
# def balance_classes(gesture_df):

#     num_classes = gesture_df.shape[0] // SAMPLE_SIZE

#     gesture_df.fillna(method='backfill', inplace=True)
#     df = np.reshape(gesture_df, (num_classes // SAMPLE_SIZE, SAMPLE_SIZE, gesture_df.shape[1]))

# #%%

# %%
