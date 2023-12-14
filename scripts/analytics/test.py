#%%


from pybaseball import playerid_lookup, statcast_pitcher


player_id = playerid_lookup('cole', 'gerrit')
id = player_id['key_mlbam'][0]
data = statcast_pitcher(start_dt="2023-03-31", end_dt="2023-04-23", player_id=id)
#data2022 = statcast_pitcher(start_dt="2022-03-30", end_dt="2022-04-05", player_id=id)
data

#%%

data_ff = data.query("pitch_type == 'FF'")


#%%


data[data['pitch_type'].isin(['FF','KC'])]


#%%




#%%




#%%