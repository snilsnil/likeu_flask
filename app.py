from flask import Flask, json, jsonify, redirect, request, render_template,  url_for
import csv
import zipfile
import os
import datetime
from user.user_class.videosMarge import VideosMarge

app = Flask(__name__)

@app.route('/ball/<id>', methods=['GET'])
def respose_ball_json(id):
    # 여기서 id와 player를 사용하여 원하는 작업을 수행합니다
    
    try:
        int(id)
        data=f'user/user_ball/{id}.json'
    except ValueError:
        data=f'basketball/basketball_ball/{id}.json'
        
    if os.path.exists(data): 
        with open(data, 'r') as f: 
            data = json.load(f) 
            return jsonify(data) 
    else: return jsonify({"error": "File not found"}),

@app.route('/player/<id>', methods=['GET'])
def respose_player_json(id):
    # 여기서 id와 player를 사용하여 원하는 작업을 수행합니다
    
    try:
        int(id)
        data=f'user/user_player/{id}.json'
    except ValueError:
        data=f'basketball/basketball_player/{id}.json'
        
    if os.path.exists(data): 
        with open(data, 'r') as f: 
            data = json.load(f) 
            return jsonify(data) 
    else: return jsonify({"error": "File not found"}),
    
@app.route('/similarity/<id>&<player>', methods=['GET'])
def similarity_json(id, player):
    # 여기서 id와 player를 사용하여 원하는 작업을 수행합니다
    
    data=f'user/dtw/result/{id}_{player}.json'
    if os.path.exists(data): 
        with open(data, 'r') as f: 
            data = json.load(f) 
            return jsonify(data) 
    else: return jsonify({"error": "File not found"}),

# 비디오 업로드 및 처리 라우트
@app.route('/upload/<player>', methods=['POST'])
def upload_video(player):
    UPLOAD_FOLDER = 'uploads'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # player=request.form.get('player')
    datetime_now_string = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if 'video' not in request.files:
        return "No video file", 400

    video = request.files['video']
    if video.filename == '':
        return "No selected video", 400
    else :
        video.filename = datetime_now_string+'.MOV'

    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)
    
    filename = video_path.split(".")[0]
    result_video_path = filename.split("/")[1]
    

    csv_filename = '{}.json'.format(datetime_now_string)
    VideosMarge(video_path, csv_filename, player)
    
    # url=[f'http:localhost:3000/ball/{result_video_path}', 
    #     f'http:localhost:3000/player/{result_video_path}',
    #     f'http:localhost:3000/ball/{player}', 
    #     f'http:localhost:3000/player/{player}',
    #     f'http:localhost:3000/similarity/{result_video_path}&{player}']
    
    return redirect(url_for('similarity_json', id=result_video_path, player=player))

# 메인 페이지 라우트
@app.route('/')
def index():
    upload_video_name = os.listdir('./uploads')
    print(upload_video_name)
    return render_template('index.html', upload_video_name=upload_video_name)

if __name__=="__main__":
    app.run(debug=True, port=3000, host='0.0.0.0')