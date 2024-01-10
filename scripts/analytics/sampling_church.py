#%%

import random
random.seed(2024)

name_list = ["凃承孝", "巫哲嘉", "林顗德","Eva", "Helen", "蔡翔任", "林家華",
             "孫維", "吳彥達", "饒又華", "張定之", "張毅", "古佳馨", "吳逸芳"]

while name_list:
    selected_name = random.choice(name_list)
    print(selected_name)
    name_list.remove(selected_name)

#%%