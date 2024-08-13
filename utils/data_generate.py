nodes = {
    'Start of journey': -2,"Home Page": 0, "Bike Models Overview":1, "Specific Bike Model Page":2, "Build & Customize":3, 
    "Compare Models":4, "Find a Dealer":5, "Request a Quote":6, "Check Financing Options":7, 
    "Schedule a Test Ride":8, "Customer Reviews":9, "Accessory Shop":10, "Service & Maintenance Info":11, 
    "User Account Login/Registration":12, "Contact Support":13, "Promotions & Offers":14, 
    "Blog/News/Updates":15, "Events & Community":16, "Bike Registration":17, "Bike Insurance Information":18, 
    "FAQ/Help Center":19,'End of Journey':-1}

import json
import sys
import pandas as pd
with open("/home/sphaire/gnn_llm/Graph-LLM/dataset/custom/LLM_generated_chat.json") as file:
    data = json.load(file)

### for next node
if sys.argv[1] == 'next_node':
    print(f'in next')
    conversations = []
    node_lis = []
    v_node_hist = []
    no_chat = 0
    for key, value in data.items():

        conversation = []
        visited_node = []
        # chats = []
        c = 0

        for p,i in enumerate(value['context']):
            conversation.append(i)
            if "User" in i.split(':')[0]:
                
                if c!= 0:
                    v_node_hist.append(['Start of journey'] + value['nodes_visited'][:c])
                    
                else:
                    v_node_hist.append(['Start of journey'])

                conversations.append('\n'.join(conversation))

                node_lis.append(value['nodes_visited'][c])

                c+=1
                no_chat+=1

### for current node
if sys.argv[1] == 'current_node':
    print('in current')
    conversations = []
    node_lis = []
    v_node_hist = []
    no_chat = 0
    for key, value in data.items():

        conversation = []
        visited_node = []
        # chats = []
        c = 0

        for p,i in enumerate(value['context']):
            conversation.append(i)
            if "User" in i.split(':')[0]:
                
                if c== 0:
                    v_node_hist.append([])
                    
                elif c==1:
                    v_node_hist.append(['Start of journey'])
                else:
                    v_node_hist.append(['Start of journey'] + value['nodes_visited'][:c-1])

                conversations.append('\n'.join(conversation))
                if c == 0:
                    node_lis.append('Start of journey')
                else:
                    node_lis.append(value['nodes_visited'][c - 1])

                c+=1
                no_chat+=1

mapp = {}
ds = []
for c,i in enumerate(conversations):
    mapp = {}
    mapp['node_ids'] = c
    mapp['node_feat'] = i + ", The list of nodes visited till now is " + str(v_node_hist[c]) + ". next node is "
    mapp['label'] = nodes[node_lis[c]]
    ds.append(mapp)


file_path = f'/home/sphaire/gnn_llm/trial/Graph-LLM/dataset/custom/input_to_model_{sys.argv[1]}.json'

with open(file_path, 'w') as f:
    json.dump(ds, f)

print(f'Data saved to {file_path}')
df = pd.read_json(file_path)
from datasets import load_dataset, Dataset
data = Dataset.from_pandas(df)
data.save_to_disk(f"/home/sphaire/gnn_llm/trial/Graph-LLM/dataset/custom/{sys.argv[1]}")