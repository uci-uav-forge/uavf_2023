import numpy as np
class AlphaFilter:
    def __init__(self, init_guess, alpha, beta):
        self._iter = 0
        self._cur_estimate = init_guess
        self._alpha = alpha
        self._beta = beta
    
    def update(self, measurement, dt=1):
        if self._iter == 0:
            self._iter += 1
            return

        next_state_x = self._cur_estimate[0] + dt * self._cur_estimate[1]
        next_state_v = self._cur_estimate[1]
        cur_estimate_x = next_state_x + self._alpha * (measurement - next_state_x)
        cur_estimate_v = next_state_v + self._beta * ((measurement - next_state_x) / dt)
        self._cur_estimate = np.array([cur_estimate_x, cur_estimate_v])
        self._iter += 1
    
    def get_cur_estimate(self):
        return self._cur_estimate
        
# filter = AlphaFilter(1000)

# data = [996, 994, 1021, 1000, 1002, 1010, 983, 971, 993, 1023]

# for m in data:
#     filter.update(m)
#     print(filter.get_cur_estimate())

init_guess = np.array([30200, 40])
filter = AlphaFilter(init_guess, 0.2, 0.1)

data = [30171, 30353, 30756, 30799, 31018, 31278, 31276, 31379, 31748, 32175]

for m in data:
    filter.update(m, 5)
    print(filter.get_cur_estimate())
