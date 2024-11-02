import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw
import numpy as np

class DTW():
    def __init__(self, user, player):

        # JSON 파일에서 데이터를 읽어옵니다
        self.df_mlb = pd.read_json(f'basketball/basketball_player/{player}.json')
        self.df_usr = pd.read_json(f'user/user_player/{user}.json')
        
        self.data_list=[]

        # 필요한 열만 선택
        self.df_mlb_filtered = self.df_mlb[self.df_mlb['Shooting'] == True][['Elbow Angle','Knee Angle']]
        self.df_usr_filtered = self.df_usr.iloc[6:22][['Elbow Angle','Knee Angle']]

        # 엘보 각도만 추출하여 배열로 변환
        self.line1 = self.df_usr_filtered[['Elbow Angle']].to_numpy()
        self.line2 = self.df_mlb_filtered[['Elbow Angle']].to_numpy()

        # DTW 거리 계산
        self.distance = dtw.distance(self.line1, self.line2)
        self.max_distance = len(self.line1) * np.max([np.max(self.line1), np.max(self.line2)])
        self.similarity_percentage = (1 - self.distance / self.max_distance) * 100

        print(f"DTW 거리: {self.distance}")
        print(f"유사도: {self.similarity_percentage:.2f}%")
        
        self.data_list.append({
            'user': user,
            'mlb': player,
            'distance':self.distance,
            "similarity_percentage":round(self.similarity_percentage)
        })
        
        df = pd.DataFrame(self.data_list)
        df.to_json(f'user/dtw/result/{user}_{player}.json', orient='records', indent=4)
    

if __name__=='__main__':
    DTW()
