####################################
#       AirSim 작동 여부 판단
# ##################################
#

# client = airsim.CarClient()
# client.confirmConnection()

# print("AirSim 연결 성공!")
# 결과>> AirSim 연결 성공 프린트 완료
#####################################

####################################
#       AirSim 데이터 수집
# ##################################
import airsim
import numpy as np
import cv2
import time

# AirSim 클라이언트 연결
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

print("AirSim 연결 성공!")

# 사용 가능한 카메라 ID 목록
camera_ids = ["0", "1"]

# 차량 움직이기 시도
print("차량 전진 시도")

# 속도 조절 (0.0 ~ 1.0)까지의 값 선택 가능
# 가장 빠른 것이 1, 느린 것이 0
car_controls.throttle = 0.5

# 회전 각도(0.0 ~ 1.0)까지의 값 선택 가능
# 0 = 직진, 수가 커질수록 회전 각도가 커짐
car_controls.steering = 0.3

client.setCarControls(car_controls)
time.sleep(3)  # 3초간 전진

# 차량 정지
print("차량 정지")
car_controls.throttle = 0
client.setCarControls(car_controls)

# 카메라 데이터 수신 테스트
print("\n사용 가능한 카메라 확인:")
for cam_id in camera_ids:
    try:
        # 이미지 요청
        responses = client.simGetImages([airsim.ImageRequest(cam_id, airsim.ImageType.Scene, False, False)])
        if responses and responses[0].width > 0 and responses[0].height > 0:
            print(f"카메라 '{cam_id}'에서 이미지 수신 성공")
            
            # 이미지 변환 및 출력
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)  # 바이트 배열을 numpy 배열로 변환
            img2d = img1d.reshape(responses[0].height, responses[0].width, 3)  # 2D 이미지로 변환
            cv2.imshow(f"AirSim Camera {cam_id}", img2d)  # OpenCV 창으로 출력
        else:
            print(f"카메라 '{cam_id}'에서 이미지 수신 실패")
    
    except Exception as e:
        print(f"카메라 '{cam_id}'에서 오류 발생: {str(e)}")

# 키 입력 대기 후 창 닫기
cv2.waitKey(0)
cv2.destroyAllWindows()

# API 제어 해제
client.enableApiControl(False)
print("AirSim API 제어 해제")
