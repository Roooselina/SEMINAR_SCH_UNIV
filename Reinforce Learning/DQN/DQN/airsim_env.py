import airsim
import numpy as np
import gym
from gym import spaces
import time

class AirSimCarEnv(gym.Env):
    def __init__(self):
        super(AirSimCarEnv, self).__init__()
        
        # AirSim 연결
        self.client = airsim.CarClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.car_controls = airsim.CarControls()

        # 상태 공간 | speed, steering, 거리 센서
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        # 행동 공간 |전진, 좌회전, 우회전
        # 실제 에이전트가 선택할 수 있는 옵션을 3개 중 하나로 고를 예정
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        """ 주어진 action에 따라 차량을 움직이고, 보상을 계산 """
        # 행동 정의
        if action == 0:  # 직진
            self.car_controls.throttle = 0.5
            self.car_controls.steering = 0.0
        elif action == 1:  # 좌회전
            self.car_controls.throttle = 0.3
            self.car_controls.steering = -0.5
        elif action == 2:  # 우회전
            self.car_controls.throttle = 0.3
            self.car_controls.steering = 0.5
        
        self.client.setCarControls(self.car_controls)
        time.sleep(0.2)  # 행동 적용 후 대기
        
        # 차량 상태 가져오기
        car_state = self.client.getCarState()
        speed = car_state.speed

        # 충돌 감지
        collision_info = self.client.simGetCollisionInfo()
        done = collision_info.has_collided  # 충돌하면 종료
        
        # 보상 함수
        # 충돌하면 큰 패널티 = 충돌하지 않고, 랜덤으로 운전하는 것을 목표로 함
        reward = speed if not done else -100  
        
        # 상태값 생성
        state = np.array([car_state.kinematics_estimated.position.x_val,
                          car_state.kinematics_estimated.position.y_val,
                          car_state.speed,
                          self.car_controls.steering], dtype=np.float32)
        
        return state, reward, done, {}

    def reset(self):
        """ 환경 초기화 (차량 리셋) """
        self.client.reset()
        self.client.enableApiControl(True)
        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.client.setCarControls(self.car_controls)
        time.sleep(1)  # 환경 리셋 대기
        return np.zeros(4, dtype=np.float32)  # 초기 상태 반환

    def close(self):
        """ 환경 종료 """
        self.client.enableApiControl(False)
