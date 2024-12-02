import cv2
import numpy as np
import math
import time
from pynput.keyboard import Key, Controller
from PIL import ImageGrab
import pygetwindow as gw
import keyboard as kb

keyboard = Controller()

stay = 0
sit_right = 0
sit_left = 0

def detect_lane_lines(frame):
    global sit_left, sit_right, stay  # sit_left와 sit_right를 전역 변수로 사용
    mid_point = None
    left_center_point = -1  # 기본값 (초기화)
    right_center_point = -1  # 기본값 (초기화)
    line_center_x = None  # 초기화 추가
    
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 가우시안 블러 적용 (노이즈 제거)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny Edge Detection (엣지 검출)
    edges = cv2.Canny(blur, 30, 120)
    
    # 4. 관심 영역(ROI) 설정
    height, width = frame.shape[:2]
    
    vertices_center = np.array([[(400, 350), (900, 350), (900, 500), (400, 500)]], dtype=np.int32)  # 중앙 ROI 설정

    # 각 ROI에 대해 마스크 설정
    mask_center = np.zeros_like(edges)
    cv2.fillPoly(mask_center, vertices_center, 255)

    # 각 ROI별로 마스크 적용
    masked_edges_center = cv2.bitwise_and(edges, mask_center)

    # 허프 변환을 이용한 직선 검출 - 각 ROI에 다른 파라미터 적용
    lines_center = cv2.HoughLinesP(masked_edges_center, 1, np.pi / 180, threshold=20, minLineLength=5, maxLineGap=120)

    # y=305 위치에 가로선 그리기
    cv2.line(frame, (0, 337), (width, 337), (0, 0, 255), 2)

    # 중앙 ROI의 선 그리기
    if lines_center is not None:
        for line in lines_center:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                angle_center = math.degrees(math.atan(slope))
                if abs(angle_center) <= 30:
                    continue
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 선의 중앙점 계산
            line_center_x = (x1 + x2) // 2

            # 왼쪽과 오른쪽에서 가장 가까운 선 찾기
            if line_center_x < center_x:
                left_center_point = line_center_x
            elif line_center_x > center_x:
                right_center_point = line_center_x



    cv2.polylines(frame, vertices_center, isClosed=True, color=(135, 206, 250), thickness=2)
    
    # 중앙 세로선 (x 좌표의 중앙 점 기준)
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 0), 2)  # 중앙 세로선

    # 파란 선들 (허프 변환 결과) 중 중앙 세로선과의 거리 계산
    closest_left_distance = float('inf')
    closest_right_distance = float('inf')
    for line in lines_center:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
            angle_center = math.degrees(math.atan(slope))
            if abs(angle_center) <= 30:
                continue
        

        
        # 중앙 세로선과의 수평 거리 계산
        distance = abs(center_x - line_center_x)
        


    # 거리 출력
    if stay == 0:
        print(f"왼쪽에서 가장 가까운 선과의 거리: {closest_left_distance} 픽셀")
        print(f"오른쪽에서 가장 가까운 선과의 거리: {closest_right_distance} 픽셀")
    

            
            
    if line_center_x is not None:
        if line_center_x < center_x:
            left_center_point = line_center_x
        elif line_center_x > center_x:
            right_center_point = line_center_x

    # 중앙점(mid_point) 계산
    if left_center_point != -1 and right_center_point != -1:
        mid_point = (left_center_point + right_center_point) // 2
    else:
        mid_point = None  # 선이 제대로 감지되지 않음

    # mid_point와 center_x를 비교하여 차선 유지 로직 수행
    if mid_point is not None:
        if (mid_point - 30) < center_x < (mid_point + 30):
            # 중앙에 가까운 경우, 거리 차이로 이동 결정
            if closest_left_distance > closest_right_distance:
                sit_left = 1  # 왼쪽으로 이동
                sit_right = 0  # 오른쪽으로 이동하지 않음
            elif closest_right_distance > closest_left_distance:
                sit_right = 1  # 오른쪽으로 이동
                sit_left = 0  # 왼쪽으로 이동하지 않음
            else:
                sit_left = 0  # 두 거리 차이가 동일한 경우
                sit_right = 0
        else:
            # center_x가 중앙점(mid_point) 범위를 벗어난 경우
            if center_x < mid_point - 30:
                print("오른쪽으로 차선 보정")
                keyboard.press(Key.right)
                time.sleep(0.02)
                keyboard.release(Key.right)
            elif center_x > mid_point + 30:
                print("왼쪽으로 차선 보정")
                keyboard.press(Key.left)
                time.sleep(0.02)
                keyboard.release(Key.left)
    else:
        print("중앙점(mid_point)을 계산할 수 없습니다. 선이 감지되지 않았을 수 있습니다.")

    # 거리 차이에 따라 추가 이동 결정
    if mid_point is None or abs(closest_left_distance - closest_right_distance) > 38:
        if closest_left_distance > closest_right_distance:
            sit_left = 2  # 왼쪽으로 크게 이동
            sit_right = 0
        elif closest_right_distance > closest_left_distance:
            sit_right = 2  # 오른쪽으로 크게 이동
            sit_left = 0
        else:
            sit_left = 0
            sit_right = 0

            

    
           
            
            

    # Return the frame with the lane markings drawn
    return frame  # Only return the frame, not edges




def capture_game_window():
    # "Euro Truck Simulator 2" 게임 창 찾기
    window = gw.getWindowsWithTitle('Euro Truck Simulator 2')
    if window:
        game_window = window[0]
        # 게임 창의 위치와 크기 가져오기
        left, top, right, bottom = game_window.left, game_window.top, game_window.right, game_window.bottom
        width = right - left
        height = bottom - top
        
        # 리사이즈할 크기 (비율 유지)
        target_width = 1280
        target_height = 768

        # 원본 비율을 계산하여 리사이즈 비율 맞추기
        resize_ratio_width = target_width / width
        resize_ratio_height = target_height / height
        resize_ratio = min(resize_ratio_width, resize_ratio_height)  # 작은 비율을 사용하여 비율 유지

        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)

        # 캡처 범위 확장 (창 크기보다 더 큰 범위로 캡처)
        extended_left = max(0, left - 1300)  # 왼쪽 확장
        extended_top = max(0, top - 50)    # 위쪽 확장
        extended_right = right + 242        # 오른쪽 확장
        extended_bottom = bottom + 157       # 아래쪽 확장

        # 확장된 영역에서 캡처
        screenshot = ImageGrab.grab(bbox=(extended_left, extended_top, extended_right, extended_bottom))
        frame = np.array(screenshot)  # PIL 이미지를 numpy 배열로 변환
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # OpenCV의 BGR 형식으로 변환

        # 리사이즈
        frame_resized = cv2.resize(frame, (new_width, new_height))  # 비율에 맞춰 리사이즈
        return frame_resized
    return None


def change_lane_right():
    global stay
    stay = 1  # 오른쪽으로 차선 변경 중
    print("오른쪽으로 차선 변경 중...")

def change_lane_left():
    global stay
    stay = 2  # 왼쪽으로 차선 변경 중
    print("왼쪽으로 차선 변경 중...")

def process_video():
    global sit_left, sit_right, stay

    while True:
        frame = capture_game_window()
        if frame is None:
            print("게임 창을 찾을 수 없습니다.")
            break  # while 루프 안쪽에 있어야 함

        # 차선 검출
        lane_frame = detect_lane_lines(frame)
        
        # 방향 조정 핫키 설정
        kb.add_hotkey('r', change_lane_right)
        kb.add_hotkey('l', change_lane_left)

        if stay == 0:
            # 왼쪽으로 이동
            if sit_left == 1:
                print("왼쪽으로 이동 중...")
                keyboard.press(Key.left)
                time.sleep(0.02)
                keyboard.release(Key.left)
                sit_left = 0  # 상태 초기화

            # 오른쪽으로 이동
            if sit_right == 1:
                print("오른쪽으로 이동 중...")
                keyboard.press(Key.right)
                time.sleep(0.02)
                keyboard.release(Key.right)
                sit_right = 0  # 상태 초기화
                
                # 왼쪽으로 이동
            if sit_left == 2:
                print("현재 중앙 영역에 있습니다. 왼쪽으로 이동 중...")
                keyboard.press(Key.left)
                time.sleep(0.04)
                keyboard.release(Key.left)
                sit_left = 0  # 상태 초기화

            # 오른쪽으로 이동
            if sit_right == 2:
                print("현재 중앙 영역에 있습니다. 오른쪽으로 이동 중...")
                keyboard.press(Key.right)
                time.sleep(0.04)
                keyboard.release(Key.right)
                sit_right = 0  # 상태 초기화

        elif stay == 1:  # 오른쪽 차선 변경
            keyboard.press(']')
            time.sleep(0.1)
            keyboard.release(']')
            time.sleep(2)
            print("오른쪽으로 차선 변경 중...")
            keyboard.press(Key.right)
            time.sleep(0.27)
            keyboard.release(Key.right)
            keyboard.press(']')
            time.sleep(0.1)
            keyboard.release(']')
            stay = 0  # 차선 변경 후 stay 값을 0으로 리셋

        elif stay == 2:  # 왼쪽 차선 변경
            keyboard.press('[')
            time.sleep(0.1)
            keyboard.release('[')
            time.sleep(2)
            print("왼쪽으로 차선 변경 중...")
            keyboard.press(Key.left)
            time.sleep(0.27)
            keyboard.release(Key.left)
            keyboard.press('[')
            time.sleep(0.1)
            keyboard.release('[')
            stay = 0  # 차선 변경 후 stay 값을 0으로 리셋

        # 결과 출력
        cv2.imshow('Lane Detection', lane_frame)  # Now it's just lane_frame

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

process_video()




