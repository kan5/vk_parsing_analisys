import json
import pandas as pd
from itertools import chain
import requests
import time
from tqdm import tqdm
import numpy as np
import datetime
import os
import networkx as nx


def get_members_df(event_id, date, data_folder='../../data/vk-graph', load_likes=True):
    
    sure_members_path = f'{data_folder}/{event_id}/{date}/sure_members.json'
    unsure_members_path = f'{data_folder}/{event_id}/{date}/unsure_members.json'
    group_info_path= f'{data_folder}/{event_id}/{date}/group_info.json'
    friends_path = f'{data_folder}/{event_id}/{date}/friends.jsonl'
    followers_path = f'{data_folder}/{event_id}/{date}/followes.jsonl'
    subscriptions_people_path = f'{data_folder}/{event_id}/{date}/sub_people.jsonl'
    subscriptions_groups_path = f'{data_folder}/{event_id}/{date}/sub_groups.jsonl'
    posts_path = f'{data_folder}/{event_id}/{date}/posts.jsonl'
    likes_path = f'{data_folder}/{event_id}/{date}/likes.jsonl'
    group_posts_path = f'{data_folder}/{event_id}/{date}/group_posts.jsonl'
    group_likes_path = f'{data_folder}/{event_id}/{date}/group_likes.jsonl'
    
    with open(sure_members_path) as f:
        sure_members = json.load(f)
    with open(group_info_path) as f:
        group_info = json.load(f)
    unsure_members = []
    if group_info.get('type') == 'event':
        with open(unsure_members_path) as f:
            unsure_members = json.load(f)

    sure_df = pd.DataFrame(sure_members)

    not_sure_df = pd.DataFrame(unsure_members)

    all_people_df = pd.concat([
        # all_people, 
       sure_df, 
       not_sure_df]).reset_index()

    use_cols = [
        'id',
        'is_closed', 
        'node_type', 
        'city', 
        # 'home_town',
        'sex', 
        # 'bdate',
        'byear',
        'occupation', 
        'occupation_type', 
        'relation',
        'alcohol',
        'inspired_by',
        'langs',
        # 'langs_full',
        'life_main',
        'people_main',
        'political',
        'religion',
        # 'religion_id',
        'smoking',
        'followers_count', 
        'is_org',
        'first_name',
        'last_name',
        'status',
    ]

    if group_info.get("contacts"):
        orgs = [i for i in group_info.get("contacts") if i.get('user_id')]
        if orgs:
            all_people_df['is_org'] = all_people_df.id.isin(list(map(lambda x: x.get('user_id'), orgs)))
        else:
            all_people_df['is_org'] = False
    else:
        all_people_df['is_org'] = False

    all_people_df['is_sure_member'] = all_people_df.id.isin(sure_df['id'])

    all_people_df['node_type'] = all_people_df.apply(lambda x: 'sure_member' if x['is_sure_member'] \
                                                   else 'unsure_member', axis=1)

    all_people_df['city'] = all_people_df['city'].map(lambda x: x.get('title'), na_action='ignore')
    all_people_df['country'] = all_people_df['country'].map(lambda x: x.get('title'), na_action='ignore')
    all_people_df['occupation_type'] = all_people_df['occupation'].map(lambda x: x.get('type'), na_action='ignore')
    all_people_df['occupation'] = all_people_df['occupation'].map(lambda x: x.get('name'), na_action='ignore')
    personals = {
        'political': {
            1: "коммунистические",
            2: "социалистические",
            3: "умеренные",
            4: "либеральные",
            5: "консервативные",
            6: "монархические",
            7: "ультраконсервативные",
            8: "индифферентные",
            9: "либертарианские",
        },
        "people_main": {
            1: "ум и креативность",
            2: "доброта и честность",
            3: "красота и здоровье",
            4: "власть и богатство",
            5: "смелость и упорство",
            6: "юмор и жизнелюбие",
        },
        "life_main": {
            1: "семья и дети",
            2: "карьера и деньги",
            3: "развлечения и отдых",
            4: "наука и исследования",
            5: "совершенствование мира",
            6: "саморазвитие",
            7: "красота и искусство",
            8: "слава и влияние",
        },
    }
    for field, dic in personals.items():
        all_people_df[field] = all_people_df['personal'].map(lambda x: dic.get(int(x.get(field)))  if x and x.get(field) else None, na_action='ignore')
    relation = {
        1: "не женат/не замужем",
        2: "есть друг/есть подруга",
        3: "помолвлен/помолвлена",
        4: "женат/замужем",
        5: "всё сложно",
        6: "в активном поиске",
        7: "влюблён/влюблена",
        8: "в гражданском браке",
        0: "не указано"
    }
    all_people_df['alcohol'] = all_people_df['personal'].map(lambda x: x.get('alcohol', np.nan) if x and not (x.get('alcohol') == 0) else np.nan, na_action='ignore')
    all_people_df['smoking'] = all_people_df['personal'].map(lambda x: x.get('smoking', np.nan) if x and not (x.get('smoking') == 0) else np.nan, na_action='ignore')
    all_people_df['relation'] = all_people_df['relation'].map(lambda x: relation.get(int(x)), na_action='ignore')
    for cat in [
        'inspired_by',
        'langs',
        'religion']:
        all_people_df[cat] = all_people_df['personal'].map(lambda x: x.get(cat) if x and x.get(cat) else None, na_action='ignore')
    all_people_df['langs'] = all_people_df['langs'].map(lambda x: ','.join(x) if isinstance(x, list) else x)
    all_people_df['sex'] = all_people_df['sex'].map({1: 'woman', 2: 'man', 0: None})
    all_people_df['byear'] = all_people_df['bdate'].map(lambda x: int(x.split('.')[-1]) if x and len(x.split('.')) == 3 else None, na_action='ignore')
    all_people_df['byear'] = all_people_df['byear'].map(lambda x: x if x > 1945 else np.nan)
    all_people_df = all_people_df[use_cols].set_index('id')
    
    all_people_df['main_group_likes'] = 0
    with open(group_likes_path) as f:
        for line in f:
            i = json.loads(line)
            for uid in i.get('likes'):
                if uid in all_people_df.index:
                    all_people_df.loc[uid, 'main_group_likes'] += 1
    
    if load_likes:
        got_likes = {}
        got_likes_from_members = {}
        with open(likes_path) as f:
            for line in f:
                i = json.loads(line)
                if i.get('user_id') not in all_people_df.index:
                        continue
                if i.get('user_id') not in got_likes:
                    got_likes[i.get('user_id')] = 0
                    got_likes_from_members[i.get('user_id')] = 0
                for uid in i.get('likes'):
                    if uid in all_people_df.index:
                        got_likes_from_members[i.get('user_id')] += 1
                        # all_people_df.loc[i.get('user_id'), 'got_likes_from_members'] += 1
                    got_likes[i.get('user_id')] += 1
                    # all_people_df.loc[i.get('user_id'), 'got_likes'] += 1

        all_people_df['got_likes_from_members'] = all_people_df.index.map(got_likes_from_members)
        all_people_df['got_likes'] = all_people_df.index.map(got_likes)
        all_people_df['got_likes'] = all_people_df.got_likes.fillna(0)
        all_people_df['got_likes_from_members'] = all_people_df.got_likes_from_members.fillna(0)
    
    all_people_df = all_people_df.fillna(np.nan)
    
    return all_people_df


def get_people_graph(event_id, date, only_members=False, load_likes=True):
    data_folder = '../../data/vk-graph'
    sure_members_path = f'{data_folder}/{event_id}/{date}/sure_members.json'
    unsure_members_path = f'{data_folder}/{event_id}/{date}/unsure_members.json'
    group_info_path= f'{data_folder}/{event_id}/{date}/group_info.json'
    friends_path = f'{data_folder}/{event_id}/{date}/friends.jsonl'
    followers_path = f'{data_folder}/{event_id}/{date}/followes.jsonl'
    subscriptions_people_path = f'{data_folder}/{event_id}/{date}/sub_people.jsonl'
    subscriptions_groups_path = f'{data_folder}/{event_id}/{date}/sub_groups.jsonl'
    posts_path = f'{data_folder}/{event_id}/{date}/posts.jsonl'
    likes_path = f'{data_folder}/{event_id}/{date}/likes.jsonl'
    
    G = nx.DiGraph()
    
    with open(group_info_path) as f:
        group_info = json.load(f)
    with open(sure_members_path) as f:
        for node in json.load(f):
            G.add_node(node.get("id"), attr_dict={**node,**{'mebership': 'sure'}})
    unsure_members = []
    if group_info.get('type') == 'event':
        with open(unsure_members_path) as f:
            for node in json.load(f):
                G.add_node(node.get("id"), attr_dict=node|{'mebership': 'unsure'})

    with open(subscriptions_people_path) as f:
        for line in f:
            i = json.loads(line)
            for u, vs in i.items():
                for j in vs:
                    if only_members and not ((int(u) in G.nodes) and (j in G.nodes)):
                        continue
                    G.add_edge(int(u), j)
    with open(friends_path) as f:
        for line in f:
            i = json.loads(line)
            for u, vs in i.items():
                for j in vs:
                    if only_members and not ((int(u) in G.nodes) and (j in G.nodes)):
                        continue
                    G.add_edge(int(u), j)
                    G.add_edge(j, int(u))
    
    nodes_attr = get_members_df(event_id, date, load_likes=load_likes).to_dict(orient = 'index')
    nx.set_node_attributes(G, nodes_attr)
    
    return G