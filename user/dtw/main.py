import pandas as pd
import matplotlib.pyplot as plt
from dtaidistance import dtw
import numpy as np

class DTW():
    def __init__(self, user, player):
        # JSON 파일에서 데이터를 읽어옵니다
        self.df_mlb = pd.read_json(f'basketball/basketball_player/{player}.json')
        self.df_usr = pd.read_json(f'user/user_player/{user}.json')

        self.data_list = []
        self.diff_list = []

        # 필요한 열만 선택
        self.df_mlb_filtered = self.df_mlb[self.df_mlb['Shooting'] == True][['Elbow Angle', 'Knee Angle']]
        self.df_usr_filtered = self.df_usr[self.df_usr['Shooting'] == True][['Elbow Angle', 'Knee Angle']]

        # 엘보 각도와 무릎 각도를 각각 배열로 변환
        self.line1_elbow = self.df_usr_filtered[['Elbow Angle']].to_numpy().flatten()
        self.line2_elbow = self.df_mlb_filtered[['Elbow Angle']].to_numpy().flatten()
        self.line1_knee = self.df_usr_filtered[['Knee Angle']].to_numpy().flatten()
        self.line2_knee = self.df_mlb_filtered[['Knee Angle']].to_numpy().flatten()

        # 엘보 각도 DTW 거리 계산
        self.distance_elbow = dtw.distance(self.line1_elbow, self.line2_elbow)

        # 무릎 각도 DTW 거리 계산
        self.distance_knee = dtw.distance(self.line1_knee, self.line2_knee)

        # 총합 DTW 거리 계산
        self.total_distance = self.distance_elbow + self.distance_knee
        self.max_distance_elbow = len(self.line1_elbow) * np.max([np.max(self.line1_elbow), np.max(self.line2_elbow)])
        self.max_distance_knee = len(self.line1_knee) * np.max([np.max(self.line1_knee), np.max(self.line2_knee)])
        self.max_distance_total = self.max_distance_elbow + self.max_distance_knee
        self.similarity_percentage_total = (1 - self.total_distance / self.max_distance_total) * 100

        print(f"엘보 각도 DTW 거리: {self.distance_elbow}")
        print(f"무릎 각도 DTW 거리: {self.distance_knee}")
        print(f"총합 DTW 거리: {self.total_distance}")
        print(f"총합 유사도: {self.similarity_percentage_total:.2f}%")

        # 팔꿈치와 무릎 각도의 차이 계산
        self.elbow_diff = [int((a - b)) for a, b in zip(self.line1_elbow, self.line2_elbow)]
        self.knee_diff = [int((a - b)) for a, b in zip(self.line1_knee, self.line2_knee)]

        self.data_list.append({
            'user': user,
            'mlb': player,
            'distance_total': self.total_distance,
            'similarity_percentage_total': round(self.similarity_percentage_total)
        })

        self.diff_list.append({
            'elbow_diff': self.elbow_diff,
            'knee_diff': self.knee_diff
        })
        
        df = pd.DataFrame(self.data_list)
        df.to_json(f'user/dtw/result/{user}_{player}.json', orient='records', indent=4)
        
        diff_df = pd.DataFrame(self.diff_list)
        diff_df.to_json(f'user/dtw/diff/{user}_{player}_diff.json', orient='records', indent=4)

if __name__ == '__main__':
    DTW('user123', 'player123')  # 여기에는 실제 사용자 ID와 선수 ID를 입력하세요
