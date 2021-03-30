import requests
from bs4 import BeautifulSoup as soup
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This part includes web scraping from Understat
base_url = 'https://understat.com/match/'
match = str(input('Please enter the match id: '))
url = base_url + match

req = requests.get(url)
parse_soup = soup(req.content, 'lxml')
scripts = parse_soup.find_all('script')

title = parse_soup.find('title').string
title = ' '.join(title.split()[:-4])

# get only shots data
strings = scripts[1].string

ind_start = strings.index("('") + 2
ind_end = strings.index("')")

json_data = strings[ind_start:ind_end]
json_data = json_data.encode('utf8').decode('unicode_escape')

data = json.loads(json_data)

# Saving the scraped data into dataframes
x = []
y = []
xg = []
team = []
minute = []
result = []
player = []
data_away = data['a']
data_home = data['h']

for index in range(len(data_home)):
    for key in data_home[index]:
        if key == 'xG':
            xg.append(data_home[index][key])
        if key == 'h_team':
            team.append(data_home[index][key])
        if key == 'minute':
            minute.append(data_home[index][key])
        if key == 'result':
            result.append(data_home[index][key])
        if key == 'player':
            player.append(data_home[index][key])

for index in range(len(data_away)):
    for key in data_away[index]:
        if key == 'xG':
            xg.append(data_away[index][key])
        if key == 'a_team':
            team.append(data_away[index][key])
        if key == 'minute':
            minute.append(data_away[index][key])
        if key == 'result':
            result.append(data_away[index][key])
        if key == 'player':
            player.append(data_away[index][key])

cols = ['minute', 'player', 'result', 'xg', 'team']
df = pd.DataFrame([minute, player, result, xg, team], index=cols)
df = df.T

# Creating a minute by minute dataframe
time = np.arange(1, 91, 1)
xg_data_home = pd.DataFrame(time)
xg_data_home.columns = ['minute']
xg_data_away = pd.DataFrame(time)
xg_data_away.columns = ['minute']

# Dividing original df into home and away and changing data types
team_lst = list(df['team'].unique())
df_home = df[df['team'] == team_lst[0]]
df_home[['minute', 'xg']] = df_home[['minute', 'xg']].apply(pd.to_numeric)
df_away = df[df['team'] == team_lst[1]]
df_away[['minute', 'xg']] = df_away[['minute', 'xg']].apply(pd.to_numeric)

# Merging the two dfs into one for both home and away
xg_data_home = pd.merge(xg_data_home, df_home, on='minute', how='left')
xg_data_home['xg'] = xg_data_home['xg'].fillna(0)
xg_data_home['team'] = xg_data_home['team'].fillna(team_lst[0])
xg_data_home['Cum xg'] = xg_data_home['xg'].cumsum()

xg_data_away = pd.merge(xg_data_away, df_away, on='minute', how='left')
xg_data_away['xg'] = xg_data_away['xg'].fillna(0)
xg_data_away['team'] = xg_data_away['team'].fillna(team_lst[1])
xg_data_away['Cum xg'] = xg_data_away['xg'].cumsum()

# Keeping a df only for goals data
home_goals = xg_data_away.loc[xg_data_away['result'] == 'Goal']
away_goals = xg_data_home.loc[xg_data_home['result'] == 'Goal']
goals = pd.concat([home_goals, away_goals])

ytick = max(xg_data_home.iloc[-1]['Cum xg'], xg_data_away.iloc[-1]['Cum xg'])
if ytick > 3:
    space = 0.5
else:
    space = 0.25
# Plotting the xG distribution graph
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(14, 10))
plt.step(xg_data_home['minute'], xg_data_home['Cum xg'], label=team_lst[0])
plt.step(xg_data_away['minute'], xg_data_away['Cum xg'], label=team_lst[1])
plt.xlabel("Minutes", fontsize=15, weight="bold")
plt.ylabel("Cumulative xG (Expected Goals)", fontsize=15, weight="bold")
plt.title(title + ' -\nxGplot (A chronological timeline of the game\'s xG narrative)', fontsize=16, weight="bold")
plt.xlim(0, 93)
plt.axvline(45, c='black')
plt.xticks(np.arange(0, 91, 15))
plt.yticks(np.arange(0, ytick, space))
texts = []
for x, y, s in zip(np.array(goals['minute']), np.array(goals['Cum xg']), goals['player']):
    texts.append(plt.text(x - 0.75, y, s, fontdict=dict(color='black', size=10, style='italic')))
plt.legend(fontsize=13)
plt.show()
