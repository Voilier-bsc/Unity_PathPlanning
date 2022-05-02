class configs:
    def __init__(self):
        #파라미터 값 세팅

        self.gpu_num = 0
        self.no_graphics = True
        self.worker_id = 1

        self.visual_state_size = [3, 32, 32]
        self.vector_state_size = 5 #agent position, target position, relative distance
        self.action_size = 2


        self.load_model = False
        self.train_mode = True

        self.batch_size = 8
        self.mem_maxlen = 50000  #replay memory 크기

        self.discount_factor = 0.9
        self.actor_lr = 1e-4
        self.critic_lr = 5e-4
        self.tau = 1e-3          #soft target update를 위한 파라미터


        # OU noise paremeters
        self.mu = 0              # 회귀할 평균의 값
        self.theta = 1e-3        # 얼마나 평균으로 빨리 회귀할 지 결정하는 값
        self.sigma = 2e-3        # 랜덤 프로세스의 변동성

        self.run_step = 50000 if self.train_mode else 0
        self.test_step = 10000
        self.train_start_step = 5000

        self.print_interval = 10
        self.save_interval = 100

        self.VISUAL_OBS = 0
        self.VECTOR_OBS = 1
        
        