
from Expert import MeanReversion, YesterdaysNews, ExpectationExpert, VolumeWeightedCloseTrend
from MutiplicativeWeights1 import MultiplicativeWeights1 as MW1
from MultiplicativeWeights2 import MultiplicativeWeights2 as MW2
from MultWeights import get_data_in_range
import numpy as np
"""
Example code for adding experts and running the multiplicative weights algorithm
"""

"""
Load the data
"""

#setup data
path_to_data = './data/stock_data.npy'
stock_data = np.load(path_to_data, allow_pickle=True).item()


starting_day = 50
max_window = min(starting_day-1, 20)
experts = list()
experts.append(YesterdaysNews())

"""
create experts considering data from various previous time frames
"""
for window in range(1, min(20,max_window)):
    e1 = MeanReversion(window, sign=1)
    e2 = ExpectationExpert(window, sign=1)
    e3 = VolumeWeightedCloseTrend(window)
    experts += [e1, e2, e3]

#add experts to multiplicative weights algorithm

mw = MW1()
# mw = MW2()
mw.add_experts(experts)
end_day = 300
temp = stock_data['MSFT']
data = get_data_in_range(temp, 0, end_day)
out = mw.alg(data, start_day=starting_day)
decisions, expert_mistakes, mistakes, reward = out
print('num experts :', len(experts))
print('reward :', reward)
print('mistakes :', mistakes)
print('percentage of mistakes', 100 * mistakes / len(decisions))
print('min of expert mistakes', np.min(expert_mistakes))



