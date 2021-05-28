#%%

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

ylim = 300

numSamples = ["256", "512", "1024"]
maxTrees = ["100", "200", "noExp"]

os.getcwd()
# %%

# ORF figures

for n in numSamples:
    
    data1 = pickle.load(open("./data/orf_cartpole_37_data_" + str(n) + "_" + str(maxTrees[0]) + ".sav", "rb"))
    data2 = pickle.load(open("./data/orf_cartpole_37_data_" + str(n) + "_" + str(maxTrees[1]) + ".sav", "rb"))
    data3 = pickle.load(open("./data/orf_cartpole_37_data_" + str(n) + "_" + str(maxTrees[2]) + ".sav", "rb"))

    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)
    mean3 = np.mean(data3, axis=0)

    ### Compare expansion vs. no expansion of trees
    plt.title("(Pole Balancing) ORF with $\eta$ = " + str(n))
    plt.ylim(0, ylim)
    plt.plot(mean1, label = "maxTrees=100 w. no expansion", color="green")
    plt.plot(mean3, label = "maxTrees=200 w. no expansion", color="orange")
    plt.plot(mean2, label = "maxTrees=200 w. expansion")
    plt.legend(loc="lower right")
    plt.savefig("./figures/cartpole/ORF_cartpole" + str(n) + ".jpg")
    plt.show()

    ### Show sd region for ORF with expansion
    plt.title("(Pole Balancing) ORF $\eta$=" + str(n) + ", maxTrees=200")
    plt.ylim(0, ylim)
    plt.plot(mean2, label = str(n))
    plt.fill_between(range(0, 1000), mean2 + data2.std(axis=0), mean2 - data2.std(axis=0), 
                    color = "grey",alpha=0.2)
    plt.savefig("./figures/cartpole/ORF_cartpole" + str(n) + "_with_sd" + ".jpg")
    plt.show()

# %%
# compare the three different $eta

temp0 = pickle.load(open("./data/orf_cartpole_37_data_256_200.sav", "rb"))
temp1 = pickle.load(open("./data/orf_cartpole_37_data_512_200.sav", "rb"))
temp2 = pickle.load(open("./data/orf_cartpole_37_data_1024_200.sav", "rb"))

mean0 = np.mean(temp0, axis=0)
mean1 = np.mean(temp1, axis=0)
mean2 = np.mean(temp2, axis=0)

plt.title("(Pole Balancing) ORF - Comparing different $\eta$")
plt.ylim(0, ylim)
plt.plot(mean0, label = "256")
plt.plot(mean1, label = "512")
plt.plot(mean2, label = "1024")
plt.legend(loc="lower right")
plt.savefig("./figures/cartpole/ORF_cartpole_diffEta.jpg")
plt.show()


#%%

# DQN figures

import numpy as np
import matplotlib.pyplot as plt
import pickle
# %%
dqn_size = ["32", "64", "128"]
dqn_lr = ["lr01", "lr005"]
lr = [0.01, 0.005]
ylim = 300

for s in dqn_size:
    data1 = pickle.load(open("./data/dqn_cartpole" + str(s) + "_" + dqn_lr[0] + ".sav", "rb"))
    data2 = pickle.load(open("./data/dqn_cartpole" + str(s) + "_" + dqn_lr[1] + ".sav", "rb"))

    mean1 = np.mean(data1, axis=0)
    mean2 = np.mean(data2, axis=0)

    sd1 = np.std(data1, axis=0)
    sd2 = np.std(data2, axis=0)

    ### Compare lr 0.01 vs. lr 0.005
    plt.title("(Pole Balancing) DQN with size " + str(s) + "x" + str(s))
    plt.ylim(0, ylim)
    plt.plot(mean1, label = r"$\alpha$ = 0.01")
    plt.plot(mean2, label = r"$\alpha$ = 0.005")
    plt.legend(loc="lower right")
    plt.savefig("./figures/cartpole/DQN_cartpole" + str(s) + ".jpg")
    plt.show()

    ### Show sd region for ORF with expansion
    plt.title("(Pole Balancing) DQN with size " + str(s) + "x" + str(s) + r", $\alpha$=0.005")
    plt.ylim(0, ylim)
    plt.plot(mean2)
    plt.fill_between(range(0, 1000), mean2 + sd2, mean2 - sd2, 
                    color = "grey",alpha=0.2)
    plt.savefig("./figures/cartpole/DQN_cartpole" + str(s) + "_with_sd" + ".jpg")
    plt.show()
# %%
### Compare different sizes

temp0 = pickle.load(open("./data/dqn_cartpole32_lr005.sav", "rb"))
temp1 = pickle.load(open("./data/dqn_cartpole64_lr005.sav", "rb"))
temp2 = pickle.load(open("./data/dqn_cartpole128_lr005.sav", "rb"))

mean0 = np.mean(temp0, axis=0)
mean1 = np.mean(temp1, axis=0)
mean2 = np.mean(temp2, axis=0)

plt.title("(Pole Balancing) DQN - Comparing different hidden layer sizes")
plt.ylim(0, ylim)
plt.plot(mean0, label = "32x32")
plt.plot(mean1, label = "64x64")
plt.plot(mean2, label = "128x128")
plt.legend(loc="lower right")
plt.savefig("./figures/cartpole/DQN_cartpole_diffSize.jpg")
plt.show()
# %%

## comparing DQN vs. ORF

dqn_data = pickle.load(open("./data/dqn_cartpole128_lr005.sav", "rb"))
orf_data = pickle.load(open("./data/orf_cartpole_37_data_256_200.sav", "rb"))

dqn_mean = np.mean(dqn_data, axis=0)
orf_mean = np.mean(orf_data, axis=0)

plt.title("(Pole Balancing) DQN vs. ORF")
plt.ylim(0, ylim)
plt.plot(orf_mean, label=r"ORF  ($\eta$=256, maxTrees=200)")
plt.plot(dqn_mean, label=r"DQN (size=128x128, $\alpha$=0.005)", linestyle='dotted')
plt.legend(loc="lower right")
plt.savefig("./figures/cartpole/cartpole_DQN_ORF.jpg")
plt.show()


#%%
## Blackjack ###


### DQN blackjack 


import numpy as np
import matplotlib.pyplot as plt
import pickle

dqn_size = ["32", "64", "128"]
dqn_lr = ["lr01", "lr005"]
lr = [0.01, 0.005]
ylim = 300

# %%
# Compare lr=0.01 vs 0.005 for each size

dqn_size = ["32", "64", "128"]

for s in dqn_size:
    dqn_data1 = pickle.load(open("./data/dqn_blackjack_" + s + "_01_mean.sav", "rb"))
    dqn_data2 = pickle.load(open("./data/dqn_blackjack_" + s + "_05_mean.sav", "rb"))

    mean1 = np.mean(dqn_data1, axis=0)
    mean2 = np.mean(dqn_data2, axis=0)
    
    plt.xlim(1,10)
    plt.ylim(-35, 0)
    plt.title("(Blackjack) DQN with hidden layer size " + str(s) + "x" + str(s))
    plt.plot(mean1, label="lr=0.01")
    plt.plot(mean2, label="lr=0.005")
    plt.legend(loc="lower right")
    plt.savefig("./figures/blackjack/dqn_blackjack_"  + str(s) + "x" + str(s) + '.jpg')
    plt.show()

#%%

### Compare the different hidden layer sizes

data1 = pickle.load(open("./data/dqn_blackjack_32_01_mean.sav", "rb"))
data2 = pickle.load(open("./data/dqn_blackjack_64_05_mean.sav", "rb"))
data3 = pickle.load(open("./data/dqn_blackjack_128_05_mean.sav", "rb"))

mean1 = np.mean(data1, axis=0)
mean2 = np.mean(data2, axis=0)
mean3 = np.mean(data3, axis=0)

plt.xlim(1,10)
plt.ylim(-35, 0)
plt.plot(mean1, label=r"size=32x32, $\alpha=$0.01")
plt.plot(mean2, label=r"size=64x64, $\alpha=$0.005")
plt.plot(mean3, label=r"size=128x128, $\alpha=$0.005")

plt.title("(Blackjack) DQN with different hidden layer sizes")
plt.legend(loc="lower right")
plt.savefig("./figures/blackjack/dqn_blackjack_diffSizes" + ".jpg")
# plt.show()

# %%

### figures for orf_blackjack

import numpy as np
import matplotlib.pyplot as plt
import pickle

orf_versions = ["32_100", "32_200", "64_100", "64_200", "128_100", "128_200"]

orf_eta = [32, 64, 128]
orf_maxTree = ["100", "200", "noExp"]

for v in orf_eta:
    data1 = pickle.load(open("./data/orf_blackjack_" + str(v) + "_" + orf_maxTree[0] + ".sav", "rb"))
    data2 = pickle.load(open("./data/orf_blackjack_" + str(v) + "_" + orf_maxTree[1] + ".sav", "rb"))
    data3 = pickle.load(open("./data/orf_blackjack_" + str(v) + "_" + orf_maxTree[2] + ".sav", "rb"))

    mean1 = np.mean(data1, axis=0) # row mean
    mean2 = np.mean(data2, axis=0) # row mean
    mean3 = np.mean(data3, axis=0) # row mean

    sd1 = np.std(data1, axis=0)
    sd2 = np.std(data2, axis=0)
    sd3 = np.std(data3, axis=0)
    
    plt.xlim(1,10)
    plt.ylim(-35, 0)
    plt.title(r"(Blackjack) ORF with $\eta=$" + str(v))
    
    plt.plot(range(11), mean1, label="maxTree="+orf_maxTree[0]+" w. no expansion", color="green") # 100 with no exp
    plt.plot(range(11), mean3, label="maxTree="+orf_maxTree[2]+" w. no expansion", color="orange") # 200 with no exp
    plt.plot(range(11), mean2, label="maxTree="+orf_maxTree[1]+" w. expansion", ) # 200 with exp
    
    
    plt.legend(loc="lower right")
    # plt.fill_between(range(11), cr_mean-cr_sd, cr_mean+cr_sd, color='grey', alpha=0.2)
    plt.xlabel("100 episodes")
    plt.ylabel("Average cumulative reward")
    plt.savefig("./figures/blackjack/orf_blackjack_" + str(v) + ".jpg")
    plt.show()


for v in orf_eta:
    data1 = pickle.load(open("./data/orf_blackjack_" + str(v) + "_" + orf_maxTree[1] + ".sav", "rb"))
    mean1 = np.mean(data1, axis=0)
    
    plt.xlim(1,10)
    plt.ylim(-35, 0)
    plt.plot(mean1, label=r"$\eta=$"+ str(v))
    plt.xlabel("100 episodes")
    plt.ylabel("Average cumulative reward")
    plt.title(r"(Blackjack) Comparing different $\eta$ (maxTree=200 w. expansion)")
    # plt.plot(mean1, label="size=" + str(s) + "x" + str(s))
    plt.legend(loc="lower right")
    plt.savefig("./figures/blackjack/orf_blackjack_diffEta" + ".jpg")
    # plt.show()


# %%

### (Blackjack) Compare the best of dqn vs. orf
### Best orf = \eta=32, maxTree=200
### Best dqn = 32x32, lr=0.01
best_dqn = pickle.load(open("./data/dqn_blackjack_32_01_mean.sav", "rb"))
best_orf = pickle.load(open("./data/orf_blackjack_32_200.sav", "rb"))

mean1 = np.mean(best_dqn, axis=0)
mean2 = np.mean(best_orf, axis=0)

sd1 = np.std(best_dqn, axis=0)
sd2 = np.std(best_orf, axis=0)

plt.title("(Blackjack) ORF vs. DQN")
plt.xlim(1,10)
plt.ylim(-35, 0)
plt.xlabel("100 episodes")
plt.ylabel("Average cumulative reward")
plt.plot(mean1, label=r"DQN (size=32x32, $\alpha$=0.01)")
plt.plot(mean2, label=r"ORF ($\eta=$32, maxTrees=200 w. expansion)")
plt.legend(loc="lower right")
# plt.savefig("./figures/blackjack/blackjack_DQN_ORF.jpg")
plt.show()

plt.xlim(1,10)
plt.ylim(-35, 0)
plt.xlabel("100 episodes")
plt.ylabel("Average cumulative reward")
plt.title(r"(Blackjack) DQN (size=32x32, $\alpha=$0.01) with sd")
plt.plot(mean1)
plt.fill_between(range(11), mean1+sd1, mean1-sd1, color="grey", alpha=0.2)
plt.legend(loc="lower right")
# plt.savefig("./figures/blackjack/dqn_blackjack_sd.jpg")
plt.show()

plt.xlim(1,10)
plt.ylim(-35, 0)
plt.xlabel("100 episodes")
plt.ylabel("Average cumulative reward")
plt.title(r"(Blackjack) ORF ($\eta=$32, maxTrees=200 w. expansion) with sd")
plt.plot(mean2)
plt.fill_between(range(11), mean2+sd2, mean2-sd2, color="grey", alpha=0.2)
plt.legend(loc="lower right")
# plt.savefig("./figures/blackjack/orf_blackjack_sd.jpg")
plt.show()

# %%


### ORF vs. DQN vs. standard Q in Blackjack

best_dqn = pickle.load(open("./data/dqn_blackjack_32_01_mean.sav", "rb"))
best_orf = pickle.load(open("./data/orf_blackjack_32_200.sav", "rb"))
std_ql = pickle.load(open('./data/QL_blackjack_data.sav', 'rb'))

mean1 = np.mean(best_dqn, axis=0)
mean2 = np.mean(best_orf, axis=0)
mean3 = np.mean(std_ql, axis=0)

sd1 = np.std(best_dqn, axis=0)
sd2 = np.std(best_orf, axis=0)
sd3 = np.std(std_ql, axis=0)

plt.title("(Blackjack) ORF vs. DQN vs. Standard Q-Learning")
plt.xlim(1,10)
plt.ylim(-35, 0)
plt.xlabel("100 episodes")
plt.ylabel("Average cumulative reward")
plt.plot(mean2, label=r"ORF ($\eta=$32, maxTrees=200 w. expansion)")
plt.plot(mean1, label=r"DQN (size=32x32, $\alpha$=0.01)", linestyle='dashed')
plt.plot(mean3, label=r"Standard Q-Learning", linestyle='dotted')
plt.legend(loc="upper left", fontsize = 9)
plt.savefig("./figures/blackjack/blackjack_DQN_ORF_QL.jpg")
plt.show()
# %%
help(plt.plot)
# %%
np.mean(best_dqn, axis=0)
np.mean(best_orf, axis=0)
np.std(best_dqn, axis=0)
np.mean(std_ql)
np.std(std_ql)
# %%


## LunarLander figures