import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

deliveries = pd.read_csv("D:/datasets/data/data/deliveries.csv/deliveries.csv")
match = pd.read_csv("D:/datasets/data/matches (1).csv")

df = deliveries.merge(match,left_on="match_id",right_on="id")
#df.columns
stadium=df["venue"].unique()
teams=df["batting_team"].unique()
#teams
df.sort_values(by='season')
df[df["bowling_team"]=='Sunrisers Hyderabad']["bowler"].unique()
bowlers={}
for i in teams:
    bowler=df[df["bowling_team"]==i]["bowler"].unique()
    bowlers[i]=bowler
wicket_deliveries=df[(df["dismissal_kind"]=="caught")|(df["dismissal_kind"]=="bowled")|(df["dismissal_kind"]=="lbw")|(df["dismissal_kind"]=='caught and bowled')|(df["dismissal_kind"]=="stumped")|(df["dismissal_kind"]=='hit wicket')|(df["dismissal_kind"]=="run out")]
#wicket_deliveries
wickets_by_bowler=df[(df["dismissal_kind"]=="caught")|(df["dismissal_kind"]=="bowled")|(df["dismissal_kind"]=="lbw")|(df["dismissal_kind"]=='caught and bowled')|(df["dismissal_kind"]=="stumped")|(df["dismissal_kind"]=='hit wicket')]
#wickets_by_bowler
bowler_wickets = {}
for i in teams:
    for j in bowlers[i]:
        num_of_wickets = wickets_by_bowler[wickets_by_bowler["bowler"] == j]["dismissal_kind"].count()
        bowler_wickets[j] = num_of_wickets

name = bowler_wickets.keys()
num_of_wickets = bowler_wickets.values()
bowler_wickets = pd.DataFrame([name, num_of_wickets]).T
bowler_wickets.rename(columns={0:"name",
                              1:"num_of_wickets"},inplace=True)
wickets_on_stadium = {}
for i in stadium:
    a = wicket_deliveries[wicket_deliveries["venue"] == i]["dismissal_kind"].count()
    wickets_on_stadium[i] = a

name = wickets_on_stadium.keys()
num_of_wickets = wickets_on_stadium.values()
wickets_on_stadium = pd.DataFrame([name, num_of_wickets]).T

wickets_on_stadium.rename(columns={0: "name",
                                   1: "num_of_wickets"}, inplace=True)
match_id=df["match_id"].unique()
d = {}
for i in stadium:
    match_played_on_stadium = match[match["venue"] == i]["venue"].count()
    d[i] = match_played_on_stadium

name = d.keys()
num_of_wickets = d.values()
match_played_on_stadium = pd.DataFrame([name, num_of_wickets]).T
match_played_on_stadium.rename(columns={0: "name",
                                        1: "match_played_on_stadium"}, inplace=True)

wickets_on_stadium["Avg_wicket"]=wickets_on_stadium["num_of_wickets"]/match_played_on_stadium["match_played_on_stadium"]
wickets_on_stadium.sort_values(by="Avg_wicket",ascending=False)
stadium_df = match_played_on_stadium.merge(wickets_on_stadium,on="name")
#stadium_df

# this shows no. of wickets taken by Bowler on particular stadium
stadium_bowler_wicket=wickets_by_bowler.groupby(by=["venue","bowler"],as_index=False).size()

stadium_bowler_wicket.rename(columns={"size":"num_of_wickets"},inplace=True)
#stadium_bowler_wicket

stadium_caught_by_players=wicket_deliveries[wicket_deliveries["dismissal_kind"]=="caught"].groupby(["venue","fielder"],as_index=False).size()
#stadium_caught_by_players

caught_by_players=wicket_deliveries[wicket_deliveries["dismissal_kind"]=="caught"]["fielder"].value_counts().to_frame()
#caught_by_players

team_caught=wicket_deliveries[wicket_deliveries["dismissal_kind"]=="caught"]["bowling_team"].value_counts().to_frame()
#team_caught

stadium_vs_teamcaught=wicket_deliveries[wicket_deliveries["dismissal_kind"]=="caught"].groupby(["venue","bowling_team"],as_index=False).size()
#stadium_vs_teamcaught

batsman=df["batsman"].unique()
non_striker=df["non_striker"].unique()
a = []
for i in batsman:
    if i not in bowler_wickets.name.values:
        a.append(i)
original_batman=a
#wicket_deliveries
batsman_wicket_stats=wicket_deliveries.groupby(["venue","player_dismissed","bowler"],as_index=False).size()

batsman_wicket_stats2=wicket_deliveries.groupby(["player_dismissed","bowler"],as_index=False).size()
batsman_wicket_stats2.rename(columns={"size":"num_of_wicket"},inplace=True)

seasonwise_bowler_wickets=wicket_deliveries.groupby(["season","bowler"],as_index=False).size()
seasonwise_bowler_wickets.rename(columns={"size":"num_of_wicket"},inplace=True)

venuewise_bowler_wickets=wicket_deliveries.groupby(["venue","bowler"],as_index=False).size()
venuewise_bowler_wickets.rename(columns={"size":"num_of_wicket"},inplace=True)
#venuewise_bowler_wickets

batsman_run_venue=df.groupby(["venue","batsman"],as_index=False)["total_runs"].sum()
#batsman_run_venue

batsman_run_season=df.groupby(["season","batsman"],as_index=False)["total_runs"].sum()
#batsman_run_season
batsman=list(df["batsman"].unique())
bowler=list(df["bowler"].unique())

finalteams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Daredevils']

finalcity=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah']

#pipe = pickle.load(open('pipe.pkl','rb'))
final_df_dict=pickle.load(open("final_df_dict.pkl","rb"))
final_df=pd.DataFrame(final_df_dict)

trf=ColumnTransformer([
    ("trf",OneHotEncoder(sparse=False,drop="first"),["batting_team","bowling_team","city"])
],remainder="passthrough")
pipe=Pipeline(steps=[
    ("step1",trf),
    ("step2",LogisticRegression(solver="liblinear"))
])
x=final_df.drop(columns="result")
y=final_df["result"]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)
pipe.fit(xtrain,ytrain)
def seasonwise_runs(i):
    data = batsman_run_season[batsman_run_season["batsman"] == i]
    x = data["season"].tolist()
    y = data["total_runs"].tolist()
    return x,y


def total_run_compare(a, b):
    data = batsman_run_season[batsman_run_season["batsman"] == a]
    x = data["season"].tolist()
    y = data["total_runs"].tolist()
    data1 = batsman_run_season[batsman_run_season["batsman"] == b]
    x1 = data1["season"].tolist()
    y1 = data1["total_runs"].tolist()

    return x,y,x1,y1

def total_wicket_compare(a, b):
    data = seasonwise_bowler_wickets[seasonwise_bowler_wickets["bowler"] == a]
    x = data["season"].tolist()
    y = data["num_of_wicket"].tolist()
    data1 = seasonwise_bowler_wickets[seasonwise_bowler_wickets["bowler"] == b]
    x1 = data1["season"].tolist()
    y1 = data1["num_of_wicket"].tolist()

    return x,y,x1,y1


def seasonwise_wickets(i):
    data = seasonwise_bowler_wickets[seasonwise_bowler_wickets["bowler"] == i]
    x = data["season"].tolist()
    y = data["num_of_wicket"].tolist()
    return x,y
delivery_df=pickle.load(open("delivery_df.pkl","rb"))
delivery=pd.DataFrame(delivery_df)

def match_progression(match_id):
    match = delivery[delivery['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[
        ['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'total_runs_x', 'crr',
         'rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0] * 100, 1)
    temp_df['win'] = np.round(result.T[1] * 100, 1)
    temp_df['end_of_over'] = range(1, temp_df.shape[0] + 1)

    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0, target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0, 10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]

    print("Target-", target)
    temp_df = temp_df[['end_of_over', 'runs_after_over', 'wickets_in_over', 'lose', 'win']]
    return temp_df, target
def analysis(venue, team1, team2):
    data = list(df[(df["batting_team"] == team1) & (df["season"] == 2015)]["batsman"].unique())
    data1 = list(df[(df["batting_team"] == team2) & (df["season"] == 2015)]["batsman"].unique())
    team1_batsman = []
    for i in range(1):
        r = []
        s = []
        batsman = []
        for i in data[0:10]:
            batsman.append(i)
            total_runs_venue = batsman_run_venue[(batsman_run_venue["batsman"] == i) & (batsman_run_venue["venue"] == venue)]["total_runs"].sum()
            total_runs_season = batsman_run_season[batsman_run_season["batsman"] == i]["total_runs"].sum()
            r.append(total_runs_venue)
            s.append(total_runs_season)
    team1_batsman.append(batsman)
    team1_batsman.append(r)
    team1_batsman.append(s)

    team2_batsman = []
    for i in range(1):
        r = []
        s = []
        batsman = []
        for i in data1[0:10]:
            batsman.append(i)
            total_runs_venue = \
            batsman_run_venue[(batsman_run_venue["batsman"] == i) & (batsman_run_venue["venue"] == venue)][
                "total_runs"].sum()
            total_runs_season = batsman_run_season[batsman_run_season["batsman"] == i]["total_runs"].sum()
            r.append(total_runs_venue)
            s.append(total_runs_season)
    team2_batsman.append(batsman)
    team2_batsman.append(r)
    team2_batsman.append(s)

    data3 = list(df[(df["bowling_team"] == team1) & (df["season"] == 2015)]["bowler"].unique())
    data4 = list(df[(df["bowling_team"] == team2) & (df["season"] == 2015)]["bowler"].unique())

    team1_bowler = []
    for i in range(1):
        r = []
        s = []
        bowler = []
        for i in data3:
            bowler.append(i)
            total_wicket_venue = venuewise_bowler_wickets[
                (venuewise_bowler_wickets["bowler"] == i) & (venuewise_bowler_wickets["venue"] == venue)][
                "num_of_wicket"].sum()
            total_wicket_season = seasonwise_bowler_wickets[seasonwise_bowler_wickets["bowler"] == i][
                "num_of_wicket"].sum()
            r.append(total_wicket_venue)
            s.append(total_wicket_season)
    team1_bowler.append(bowler)
    team1_bowler.append(r)
    team1_bowler.append(s)

    team2_bowler = []
    for i in range(1):
        r = []
        s = []
        bowler = []
        for i in data4:
            bowler.append(i)
            total_wicket_venue = venuewise_bowler_wickets[
                (venuewise_bowler_wickets["bowler"] == i) & (venuewise_bowler_wickets["venue"] == venue)][
                "num_of_wicket"].sum()
            total_wicket_season = seasonwise_bowler_wickets[seasonwise_bowler_wickets["bowler"] == i][
                "num_of_wicket"].sum()
            r.append(total_wicket_venue)
            s.append(total_wicket_season)
    team2_bowler.append(bowler)
    team2_bowler.append(r)
    team2_bowler.append(s)

    #         for i in range(1):
    #             x5=plt.pie(s,labels=bowler,autopct="%0.1f%%",radius=1.5)
    #             #plt.legend()
    #             plt.show()

    return team1_batsman, team2_batsman, team1_bowler, team2_bowler


##streamlit code

st.title("Team Analysis")
rad=st.sidebar.radio("Navigator",["Players Comparision","Teams Analysis","Win Predictor","Match Progression"])

if rad=="Players Comparision":
    st.subheader("Players Comparision")
    result=st.radio("Select Comparision Between :",["Batsman","Bowlers"])
    if result=="Batsman":
        user_choice=st.selectbox("select the comparision",["Univariate Analysis","Bivariate Analysis"])
        if user_choice=="Univariate Analysis":
            user_choice_batsman=st.selectbox("Select Batsman",batsman)
            x,y=seasonwise_runs(user_choice_batsman)

            plt.bar(x=x, height=y, label=user_choice_batsman)

            plt.xlabel("season")
            plt.ylabel("runs")
            plt.title("total_run")
            plt.legend()
            st.pyplot(plt)

        if user_choice=="Bivariate Analysis":
            user_choice_batsman=st.multiselect("select two batsman",batsman)
            if st.button("show comparision"):
                x,y,x1,y1=total_run_compare(user_choice_batsman[0],user_choice_batsman[1])
                col1,col2=st.columns(2)
                with col1:
                    st.write("Comparision Between Batsmans")
                    width = 0.4
                    p = np.arange(len(x))
                    p1 = [j+width for j in p]
                    plt.figure(figsize=(18, 8))
                    plt.bar(x=p, height=y,width=width,  color="r", label=user_choice_batsman[0])
                    plt.bar(x=p1, height=y1,width=width, color="b", label=user_choice_batsman[1])

                    plt.xticks(p + width, x)
                    plt.xlabel("season")
                    plt.ylabel("runs")
                    plt.title("total_run")
                    plt.legend()
                    st.pyplot(plt)

    if result=="Bowlers":
        user_choice=st.selectbox("select the comparision",["Univariate Analysis","Bivariate Analysis"])
        if user_choice=="Univariate Analysis":
            user_choice_bowler=st.selectbox("Select Bowler",bowler)
            x,y=seasonwise_wickets(user_choice_bowler)

            plt.bar(x=x, height=y, label=user_choice_bowler)

            plt.xlabel("season")
            plt.ylabel("wickets")
            plt.title("total_wickets")
            plt.legend()
            st.pyplot(plt)

        if user_choice=="Bivariate Analysis":
            user_choice_bowler = st.multiselect("select two batsman", bowler)
            if st.button("show comparision"):
                x, y, x1, y1 = total_wicket_compare(user_choice_bowler[0], user_choice_bowler[1])
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Comparision Between Bowlers")
                    # width = 0.4
                    # p = np.arange(len(x))
                    # p1 = [j + width for j in p]
                    # plt.figure(figsize=(18, 8))
                    plt.bar(x=x, height=y, color="r", label=user_choice_bowler[0])
                    plt.bar(x=x1, height=y1, color="b", label=user_choice_bowler[1])

                    # plt.xticks(p + width, x)
                    plt.xlabel("season")
                    plt.ylabel("wickets")
                    plt.title("total_wickets")
                    plt.legend()
                    st.pyplot(plt)

if rad =="Teams Analysis":
    st.subheader("Compare Stats of Team Players")
    venue = st.selectbox("Select the match venue", stadium)
    teams = st.multiselect("select the Two teams", teams)
    if st.button("Show Analysis"):
        x1,x2,x3,x4=analysis(venue,teams[0],teams[1])
        col1,col2=st.columns(2)
        with col1:
            st.write(teams[0]," Batting Performance ")
            batsman=x1[0]
            r=x1[1]
            s=x1[2]

            width=0.4
            p=np.arange(len(r))
            p1=[j+width for j in p]

            plt.figure(figsize=(18,8))
            plt.bar(x=p,height=r,width=width,color="r",label="Venue Runs")
            plt.bar(x=p1,height=s,width=width,color="b",label="Season Runs")

            plt.xticks(p+width,batsman)
            plt.xlabel("batsman")
            plt.ylabel("runs")
            plt.title("Batting Performance of the Players")
            plt.legend()
            st.pyplot(plt)

        with col2:
            st.write(teams[0],"Bowling Performance ")
            bowler=x3[0]
            r=x3[1]
            s=x3[2]
            width=0.4
            p=np.arange(len(r))
            p1=[j+width for j in p]

            plt.figure(figsize=(18,8))
            plt.bar(x=p,height=r,width=width,color="r",label="Venue Wickets")
            plt.bar(x=p1,height=s,width=width,color="b",label="Season Wickets")

            plt.xticks(p+width,bowler)
            plt.title("Bowling Performance of the Players")
            plt.xlabel("bowler")
            plt.ylabel("wickets")
            plt.legend()
            st.pyplot(plt)
        col3,col4=st.columns(2)
        with col3:
            st.write(teams[1], " Batting Performance ")
            batsman = x2[0]
            r = x2[1]
            s = x2[2]

            width = 0.4
            p = np.arange(len(r))
            p1 = [j + width for j in p]

            plt.figure(figsize=(18, 8))
            plt.bar(x=p, height=r, width=width, color="r", label="Venue Runs")
            plt.bar(x=p1, height=s, width=width, color="b", label="Season Runs")

            plt.xticks(p + width, batsman)
            plt.xlabel("batsman")
            plt.ylabel("runs")
            plt.title("Batting Performance of the Players")
            plt.legend()
            st.pyplot(plt)

        with col4:
            st.write(teams[1],"Bowling Performance ")
            bowler=x4[0]
            r=x4[1]
            s=x4[2]
            width=0.4
            p=np.arange(len(r))
            p1=[j+width for j in p]

            plt.figure(figsize=(18,8))
            plt.bar(x=p,height=r,width=width,color="r",label="Venue Wickets")
            plt.bar(x=p1,height=s,width=width,color="b",label="Season Wickets")

            plt.xticks(p+width,bowler)
            plt.title("Bowling Performance of the Players")
            plt.xlabel("bowler")
            plt.ylabel("wickets")
            plt.legend()
            st.pyplot(plt)

if rad=="Win Predictor":
    st.subheader("Win Predictor")
    col1,col2=st.columns(2)

    with col1:
        batting_team=st.selectbox("select the batting team",finalteams)
    with col2:
        bowling_team=st.selectbox("select the bowling team",finalteams)
    selected_city=st.selectbox("select city",finalcity)
    target=st.number_input("Target")
    col3,col4,col5=st.columns(3)

    with col3:
        score=st.number_input("Score")
    with col4:
        overs=st.number_input("Overs Completed")
    with col5:
        wickets=st.number_input("Wickets")

    if st.button("Predict Probability"):
        runs_left=target-score
        balls_left=120-(overs*6)
        wickets=10-wickets
        crr=score/overs
        rrr=(runs_left*6)/balls_left

        input_df=pd.DataFrame({"batting_team":[batting_team],"bowling_team":[bowling_team],
                      "city":[selected_city],"runs_left":[runs_left],"balls_left":[balls_left],
                      "wickets":[wickets],"total_runs_x":[target],"crr":[crr],"rrr":[rrr]})

        st.table(input_df)

        result = pipe.predict_proba(input_df)
        result=result*100
        st.write(bowling_team,"--",np.round(result[0][0],0),"%")
        st.write(batting_team,"--", np.round(result[0][1],0),"%")

if rad=="Match Progression":
    st.subheader("Match Progression")
    match_id=st.selectbox("select the match code",list(delivery["match_id"].unique()))
    if st.button("Show Progression"):
        temp_df,target=match_progression(match_id)
        plt.figure(figsize=(18, 8))
        plt.plot(temp_df['end_of_over'], temp_df['wickets_in_over'], color='yellow', linewidth=3)
        plt.plot(temp_df['end_of_over'], temp_df['win'], color='#00a65a', linewidth=4)
        plt.plot(temp_df['end_of_over'], temp_df['lose'], color='red', linewidth=4)
        plt.bar(temp_df['end_of_over'], temp_df['runs_after_over'])
        plt.title('Target-' + str(target))
        st.pyplot(plt)












