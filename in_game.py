import cv2
import numpy as np
import math
import time
from pynput.keyboard import Key, Controller
from PIL import ImageGrab
import pygetwindow as gw

keyboard = Controller()

# 차선 검출 함수
def detect_lane_lines(frame):
    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 가우시안 블러 적용 (노이즈 제거)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny Edge Detection (엣지 검출)
    edges = cv2.Canny(blur, 30, 120)
    
    # 4. 관심 영역(ROI) 설정
    height, width = frame.shape[:2]
    vertices1 = np.array([[(55, 100), (148, 100), (148, 342), (55, 342)]], dtype=np.int32)  # 왼쪽 ROI 설정
    vertices2 = np.array([[(1075, 100), (1180, 100), (1180, 342), (1075, 342)]], dtype=np.int32)  # 오른쪽 ROI 설정
    vertices_center = np.array([[(350, 350), (550, 350), (700, 500), (300, 500)]], dtype=np.int32)  # 중앙 ROI 설정

    # 각 ROI에 대해 마스크 설정
    mask_left = np.zeros_like(edges)
    mask_right = np.zeros_like(edges)
    mask_center = np.zeros_like(edges)
    cv2.fillPoly(mask_left, vertices1, 255)
    cv2.fillPoly(mask_right, vertices2, 255)
    cv2.fillPoly(mask_center, vertices_center, 255)

    # 각 ROI별로 마스크 적용
    masked_edges_left = cv2.bitwise_and(edges, mask_left)
    masked_edges_right = cv2.bitwise_and(edges, mask_right)
    masked_edges_center = cv2.bitwise_and(edges, mask_center)

    # 허프 변환을 이용한 직선 검출 - 각 ROI에 다른 파라미터 적용
    lines_left = cv2.HoughLinesP(masked_edges_left, 1, np.pi / 180, threshold=50, minLineLength=5, maxLineGap=100)
    lines_right = cv2.HoughLinesP(masked_edges_right, 1, np.pi / 180, threshold=50, minLineLength=5, maxLineGap=100)
    lines_center = cv2.HoughLinesP(masked_edges_center, 1, np.pi / 180, threshold=20, minLineLength=5, maxLineGap=120)

    # 각 ROI에서 y=305일 때 x 좌표 값을 저장할 리스트
    x_coords_left = []
    x_coords_right = []

    # 왼쪽 ROI의 선 그리기 및 y=305일 때 x 좌표 계산
    if lines_left is not None:
        for line in lines_left:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                angle1 = math.degrees(math.atan(slope))
                if -45 <= angle1 <= -25:
                    continue
                if abs(slope) > 0.01:
                    x_at_305_left = int((337 - y1) / slope + x1)
                else:
                    x_at_305_left = x1
                if x_at_305_left != 120:
                    x_coords_left.append(x_at_305_left)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 오른쪽 ROI의 선 그리기 및 y=305일 때 x 좌표 계산
    if lines_right is not None:
        for line in lines_right:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = (y2 - y1) / (x2 - x1)
                angle2 = math.degrees(math.atan(slope))
                if 25 <= angle2 <= 45:
                    continue
                if abs(slope) > 0.01:
                    x_at_305_right = int((337 - y1) / slope + x1)
                else:
                    x_at_305_right = x1
                if x_at_305_right != 962:
                    x_coords_right.append(x_at_305_right)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

    # x 좌표 중 가장 큰 값과 작은 값을 각각 출력
    max_x_left = max(x_coords_left) if x_coords_left else None
    min_x_right = min(x_coords_right) if x_coords_right else None

    if max_x_left:
        print(f"Left ROI: y=305일 때 가장 큰 x 좌표는 {max_x_left}")
    if min_x_right:
        print(f"Right ROI: y=305일 때 가장 작은 x 좌표는 {min_x_right}")
    cv2.polylines(frame, vertices1, isClosed=True, color=(135, 206, 250), thickness=2)
    cv2.polylines(frame, vertices2, isClosed=True, color=(135, 206, 250), thickness=2)
    cv2.polylines(frame, vertices_center, isClosed=True, color=(135, 206, 250), thickness=2)
    return frame, edges, max_x_left, min_x_right

# 게임 창 캡처 함수
# 게임 창 캡처 함수
# 게임 창 캡처 함수
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



# 비디오 처리 함수
def process_video():
    sit_right = 0
    sit_left = 0

    while True:
        frame = capture_game_window()
        if frame is None:
            print("게임 창을 찾을 수 없습니다.")
            break  # while 루프 안쪽에 있어야 함

        # 차선 검출
        lane_frame, edge, max_x_left, min_x_right = detect_lane_lines(frame)

        # 중앙 제어 방향 결정 및 이동 명령 호출
        if max_x_left and (148 - max_x_left) <= 40 and (148 - max_x_left > 0):
            sit_right = 1
            print("move right")
        elif min_x_right and (min_x_right - 1075) <= 40 and (min_x_right - 1075 > 0):
            sit_left = 1
            print("move left")
        else:
            print("here is center")
            sit_left = 0
            sit_right = 0

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

        # 결과 출력q
        cv2.imshow('Lane Detection', lane_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

process_video()
