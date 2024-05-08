from flask import Flask,request,render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictvalue', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            release_clause=float(request.form.get("release_clause")),
            best_overall=float(request.form.get("best_overall")),
            potential=float(request.form.get("potential")),
            overall_rating=float(request.form.get("overall_rating")),
            wage=float(request.form.get("wage")),
            age=float(request.form.get("age")),
            finishing=float(request.form.get("finishing")),
            composure=float(request.form.get("composure")),
            total_movement=float(request.form.get("total_movement")),
            sprint_speed=float(request.form.get("sprint_speed")),
            international_reputation=float(request.form.get("international_reputation")),
            best_position=request.form.get("best_position"),
            # Additional fields with default values of 0
            # Height=float(request.form.get("Height", 0)),
            # Weight=float(request.form.get("Weight", 0)),
            # Growth=float(request.form.get("Growth", 0)),
            # Total_attacking=float(request.form.get("Total_attacking", 0)),
            # Crossing=float(request.form.get("Crossing", 0)),
            # Heading_accuracy=float(request.form.get("Heading_accuracy", 0)),
            # Short_passing=float(request.form.get("Short_passing", 0)),
            # Volleys=float(request.form.get("Volleys", 0)),
            # Total_skill=float(request.form.get("Total_skill", 0)),
            # Dribbling=float(request.form.get("Dribbling", 0)),
            # Curve=float(request.form.get("Curve", 0)),
            # FK_Accuracy=float(request.form.get("FK_Accuracy", 0)),
            # Long_passing=float(request.form.get("Long_passing", 0)),
            # Ball_control=float(request.form.get("Ball_control", 0)),
            # Acceleration=float(request.form.get("Acceleration", 0)),
            # Agility=float(request.form.get("Agility", 0)),
            # Reactions=float(request.form.get("Reactions", 0)),
            # Balance=float(request.form.get("Balance", 0)),
            # Total_power=float(request.form.get("Total_power", 0)),
            # Shot_power=float(request.form.get("Shot_power", 0)),
            # Jumping=float(request.form.get("Jumping", 0)),
            # Stamina=float(request.form.get("Stamina", 0)),
            # Strength=float(request.form.get("Strength", 0)),
            # Long_shots=float(request.form.get("Long_shots", 0)),
            # Total_mentality=float(request.form.get("Total_mentality", 0)),
            # Aggression=float(request.form.get("Aggression", 0)),
            # Interceptions=float(request.form.get("Interceptions", 0)),
            # Att_Position=float(request.form.get("Att_Position", 0)),
            # Vision=float(request.form.get("Vision", 0)),
            # Penalties=float(request.form.get("Penalties", 0)),
            # Total_defending=float(request.form.get("Total_defending", 0)),
            # Defensive_awareness=float(request.form.get("Defensive_awareness", 0)),
            # Standing_tackle=float(request.form.get("Standing_tackle", 0)),
            # Sliding_tackle=float(request.form.get("Sliding_tackle", 0)),
            # Total_goalkeeping=float(request.form.get("Total_goalkeeping", 0)),
            # Total_stats=float(request.form.get("Total_stats", 0)),
            # Base_stats=float(request.form.get("Base_stats", 0)),
            # Pace_Diving=float(request.form.get("Pace_Diving", 0)),
            # Shooting_Handling=float(request.form.get("Shooting_Handling", 0)),
            # Passing_Kicking=float(request.form.get("Passing_Kicking", 0)),
            # Dribbling_Reflexes=float(request.form.get("Dribbling_Reflexes", 0)),
            # Defending_Pace=float(request.form.get("Defending_Pace", 0)),
            # GK_Kicking=float(request.form.get("GK_Kicking", 0)),
            # GK_Positioning=float(request.form.get("GK_Positioning", 0)),
            # GK_Handling=float(request.form.get("GK_Handling", 0)),
            # GK_Reflexes=float(request.form.get("GK_Reflexes", 0)),
            # GK_Diving=float(request.form.get("GK_Diving", 0)),
            # foot=float(request.form.get("foot", 0))
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)