# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
WORKSPACEFOLDER = 'E:\Projects\ComplexNetwork'
import os
os.chdir(WORKSPACEFOLDER)

# %% [markdown]
# # SNAPTwitter
# 
# -   cn20
# -   beta=0.0120

# %%
import json

import torch

import config


# %%
DB = config.get_DB()


# %%
with open(os.path.join(DB, 'SNAPTwitter/nodes_degree_approx_20.json'), 'r') as f:
    nodes = json.load(f)



# %%
