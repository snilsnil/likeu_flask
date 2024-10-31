import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw
import numpy as np

df_mlb = pd.read_csv("data/Booker.csv")
df_usr = pd.read_csv("data/20241031234203.csv")


df_mlb = df_mlb[df_mlb['Shooting'] == True][['Left Elbow Angle', 'Right Elbow Angle', 'Left Knee Angle', 'Right Knee Angle']]

df_usr = df_usr.iloc[6:22]
df_usr = df_usr[['elbow_angle','knee_angle']]

line1 = df_usr[['elbow_angle']].to_numpy()
line2 = df_mlb[['Left Elbow Angle']].to_numpy()

distance = dtw.distance(line1, line2)
max_distance = len(line1) * np.max([np.max(line1), np.max(line2)])
similarity_percentage = (1 - distance / max_distance) * 100

print(f"DTW 거리: {distance}")
print(f"유사도: {similarity_percentage:.2f}%")