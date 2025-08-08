import streamlit as st
import datetime
import pandas as pd
from catboost import CatBoostClassifier

model = CatBoostClassifier()
model.load_model(r"C:\Users\KRISH GUPTA\Desktop\github\Tennis-win-predictor\src\tennis_model.cbm")

st.title("_Tennis match predictor_ :tennis:")

st.header("Enter details",divider=True)
st.write("**For rank and points please refer to real time data**" )

col1,col2 = st.columns(2)

player_df = pd.read_excel(r"C:\Users\KRISH GUPTA\Desktop\github\Tennis-win-predictor\src\data\PLAYER_INFO.xlsx")
player_names = player_df['NAME']


with col1:
    player1 = st.selectbox("Player 1",player_names)
    player1_age = st.slider("Player 1 age",min_value=14,max_value=50,value=28)
    player1_rank = st.number_input("Current ranking of player 1",min_value=1,max_value=2400)
    player1_points = st.number_input("Rank points of player 1",min_value=0,max_value=15000,value=500)
with col2:
    player2 = st.selectbox("Player 2",player_names)
    player2_age = st.slider("Player 2 age",min_value=14,max_value=50,value=28)
    player2_rank = st.number_input("Current ranking of 2",min_value=1,max_value=2400)
    player2_points = st.number_input("Rank points of player 2",min_value=0,max_value=15000,value=500)

tourney_level_map = {
    "Grand Slam": "G",
    "Davis Cup": "D",
    "Futures": "F",
    "ATP 500": "A",
    "ATP Masters 1000": "M",
    "Olympics": "O"
}
level_full = st.selectbox("Select tournament level",list(tourney_level_map.keys()))
tourney_level = tourney_level_map[level_full]


surface = st.selectbox("Surface Type",("Hard","Clay","Grass"))

match_date = st.date_input("Match date")

date = int(match_date.strftime("%Y%m%d"))


p1_info = player_df.loc[player_df["NAME"]==player1].iloc[0]
player1_id = p1_info["ID"]
player1_ht = p1_info["PLAYER HT"]
player1_hand = p1_info["PLAYER HAND"]

p2_info = player_df.loc[player_df["NAME"]==player2].iloc[0]
player2_id = p2_info["ID"]
player2_hand = p2_info["PLAYER HAND"]
player2_ht = p2_info["PLAYER HT"]



input_df = pd.DataFrame([{
    'player1_id':player1_id ,
    'player2_id':player2_id ,
    'player1_hand': player1_hand,
    'player2_hand': player2_hand,
    'player1_ht': player1_ht,
    'player2_ht': player2_ht,
    'player1_age': player1_age,
    'player2_age': player2_age,
    'player1_rank': player1_rank,
    'player2_rank': player2_rank,
    'player1_rank_points': player1_points,
    'player2_rank_points': player2_points,
    'surface': surface,
    'tourney_level': tourney_level,
    'tourney_date': date

}])


st.markdown("###")

if st.button('Predict winner',use_container_width=True):
    prediction = model.predict(input_df)[0]
    if prediction==1:
        st.success(f"🎾{player1} is predicted to win!")

    else:
        st.success(f"🎾{player2} is predicted to win!")






