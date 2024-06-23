import pickle
import streamlit as st
import numpy as np

# Loading the trained model from a pickle file
model = pickle.load(open("bestmodel.pkl", "rb"))

# Defining the Streamlit app
def main():
    st.title("Sports Player Rating Predictor")

    # Creating input fields for the features
    st.write("Enter feature values for prediction:")
    movement_reactions = st.number_input("Movement Reactions", min_value=0, max_value=100, value=0)
    passing = st.number_input("Passing", min_value=0, max_value=100, value=0)
    mentality_composure = st.number_input("Mentality Composure", min_value=0, max_value=100, value=0)
    wage_eur = st.number_input("Wage EUR", min_value=0, max_value=100, value=0)
    dribbling = st.number_input("Dribbling", min_value=0, max_value=100, value=0)
    value_eur = st.number_input("Value EUR", min_value=0, max_value=100, value=0)
    attacking_sp = st.number_input("Attacking Short Passing", min_value=0, max_value=100, value=0)
    mentality_vision = st.number_input("Mentality Vision", min_value=0, max_value=100, value=0)
    potential = st.number_input("Potential", min_value=0, max_value=100, value=0)

    try:
        # Creating an array with user inputs
        X = np.array([
            movement_reactions, passing, mentality_composure, wage_eur, dribbling, value_eur,attacking_sp,mentality_vision,potential
        ]).astype(float).reshape(1, -1)

        if st.button("Predict Overall Rating"):
            prediction = model.predict(X)  # Making a prediction
            st.write(f"Overall Rating Prediction: {prediction[0]:.2f}")

            std_prediction = np.std(prediction)
            mean_prediction = np.mean(prediction)
            confidence_score = std_prediction / mean_prediction  # Calculating confidence score

    except Exception as e:
        st.write(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
