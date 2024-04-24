import cv2
import mediapipe as mp
import numpy as np

# 제스처 정의
max_num_hands = 2
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

# MediaPipe 손 인식 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# KNN 모델 학습 데이터 불러오기
file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
angle = file[:, :-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

# 카메라 캡처 시작
cap = cv2.VideoCapture(0)

game_state = "start"
global prev_winner 
prev_winner = None

def judge_muk_jji_ppa(player1_gesture, player2_gesture):
    global prev_winner
    win_conditions = {'rock': 'scissors', 'scissors': 'paper', 'paper': 'rock'}
    
    # 이전 승자가 없는 경우 (게임의 시작 상태)
    if prev_winner is None:
        if player1_gesture == player2_gesture:
            return "Draw", None
        elif win_conditions[player1_gesture] == player2_gesture:
            prev_winner = "Left"
            return "Left", None
        else:
            prev_winner = "Right"
            return "Right", None
    # 이전 승자가 있는 경우 (게임의 진행 상태)
    else:
        # 같은 손 모양이 나온 경우, 이전 승자가 최종 승자
        if player1_gesture == player2_gesture:
            return f"{prev_winner} Final Win!", prev_winner
        # 다른 손 모양이 나온 경우, 새로운 라운드의 승자 결정
        elif win_conditions[player1_gesture] == player2_gesture:
            prev_winner = "Left"
            return "Left", None
        else:
            prev_winner = "Right"
            return "Right", None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rps_result = []
    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]
            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
            angle = np.arccos(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :], v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)
            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'gesture': rps_gesture[idx],
                    'org': org
                })

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
   
        if len(rps_result) == 2:
            player1_gesture = rps_result[0]['gesture']
            player2_gesture = rps_result[1]['gesture']
            winner_text, winner = judge_muk_jji_ppa(player1_gesture, player2_gesture)
            
            # if winner is not None:
            #     game_state = "continue"
            #     prev_winner = winner
            # else:
            #     game_state = "start"
            #     prev_winner = None
                
            cv2.putText(img, winner_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        
        cv2.imshow('Game', img)
        if cv2.waitKey(1) == ord('q'):
            break
