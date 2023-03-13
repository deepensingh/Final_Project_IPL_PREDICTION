import streamlit as st
import pickle
import pandas as pd

# Loading data and model
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore', 'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings', 'Rajasthan Royals', 'Delhi Capitals']
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']
gbm = pickle.load(open('gbm.pkl','rb'))

# Set page configuration
st.set_page_config(page_title='IPL Win Predictor', page_icon=':bar_chart:', layout='wide', initial_sidebar_state='auto')

# Set background color
st.markdown("""
<style>
body {
    background-color: #f6f5f5;
}
</style>
""", unsafe_allow_html=True)

# Set app title
st.title('IPL Win Predictor')

# Add a space on top of the page for an image
from PIL import Image

image = Image.open('img.jpg')
resized_image = image.resize((500, 500))
st.image(resized_image, width=500)

# Create dropdowns for input selection
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

# Create numeric input fields for target, score, overs and wickets
target = st.number_input('Target', value=0, step=1, format='%d')
col3,col4,col5 = st.columns(3)
with col3:
    score = st.number_input('Score', step=1, value=0)
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets = st.number_input('Wickets out', step=1, value=0)

# Create a button to trigger prediction
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    # Create input dataframe for model
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})

    # Use model to predict probability of win or loss
    result = gbm.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    # Show output as percentage
    st.header(batting_team + "- " + str(round(win*100)) + "%")
    st.header(bowling_team + "- " + str(round(loss*100)) + "%")
