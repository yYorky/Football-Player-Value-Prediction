import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                 release_clause: int,
                 best_overall: int,
                 potential: int,
                 overall_rating: int,
                 wage: int,
                 age: int,
                 finishing: int,
                 composure: int,
                 total_movement: int,
                 sprint_speed: int,
                 international_reputation: int,
                 best_position: str,
                 # Additional fields with default values of 0
                 Height: int = 0,
                 Weight: int = 0,
                 Growth: int = 0,
                 Total_attacking: int = 0,
                 Crossing: int = 0,
                 Heading_accuracy: int = 0,
                 Short_passing: int = 0,
                 Volleys: int = 0,
                 Total_skill: int = 0,
                 Dribbling: int = 0,
                 Curve: int = 0,
                 FK_Accuracy: int = 0,
                 Long_passing: int = 0,
                 Ball_control: int = 0,
                 Acceleration: int = 0,
                 Agility: int = 0,
                 Reactions: int = 0,
                 Balance: int = 0,
                 Total_power: int = 0,
                 Shot_power: int = 0,
                 Jumping: int = 0,
                 Stamina: int = 0,
                 Strength: int = 0,
                 Long_shots: int = 0,
                 Total_mentality: int = 0,
                 Aggression: int = 0,
                 Interceptions: int = 0,
                 Att_Position: int = 0,
                 Vision: int = 0,
                 Penalties: int = 0,
                 Total_defending: int = 0,
                 Defensive_awareness: int = 0,
                 Standing_tackle: int = 0,
                 Sliding_tackle: int = 0,
                 Total_goalkeeping: int = 0,
                 Total_stats: int = 0,
                 Base_stats: int = 0,
                 Pace_Diving: int = 0,
                 Shooting_Handling: int = 0,
                 Passing_Kicking: int = 0,
                 Dribbling_Reflexes: int = 0,
                 Defending_Pace: int = 0,
                 GK_Kicking: int = 0,
                 GK_Positioning: int = 0,
                 GK_Handling: int = 0,
                 GK_Reflexes: int = 0,
                 GK_Diving: int = 0,
                 foot: int = 0):

        self.release_clause = release_clause
        self.best_overall = best_overall
        self.potential = potential
        self.overall_rating = overall_rating
        self.wage = wage
        self.age = age
        self.finishing = finishing
        self.composure = composure
        self.total_movement = total_movement
        self.sprint_speed = sprint_speed
        self.international_reputation = international_reputation
        self.best_position = best_position
        # Additional fields
        self.Height = Height
        self.Weight = Weight
        self.Growth = Growth
        self.Total_attacking = Total_attacking
        self.Crossing = Crossing
        self.Heading_accuracy = Heading_accuracy
        self.Short_passing = Short_passing
        self.Volleys = Volleys
        self.Total_skill = Total_skill
        self.Dribbling = Dribbling
        self.Curve = Curve
        self.FK_Accuracy = FK_Accuracy
        self.Long_passing = Long_passing
        self.Ball_control = Ball_control
        self.Acceleration = Acceleration
        self.Agility = Agility
        self.Reactions = Reactions
        self.Balance = Balance
        self.Total_power = Total_power
        self.Shot_power = Shot_power
        self.Jumping = Jumping
        self.Stamina = Stamina
        self.Strength = Strength
        self.Long_shots = Long_shots
        self.Total_mentality = Total_mentality
        self.Aggression = Aggression
        self.Interceptions = Interceptions
        self.Att_Position = Att_Position
        self.Vision = Vision
        self.Penalties = Penalties
        self.Total_defending = Total_defending
        self.Defensive_awareness = Defensive_awareness
        self.Standing_tackle = Standing_tackle
        self.Sliding_tackle = Sliding_tackle
        self.Total_goalkeeping = Total_goalkeeping
        self.Total_stats = Total_stats
        self.Base_stats = Base_stats
        self.Pace_Diving = Pace_Diving
        self.Shooting_Handling = Shooting_Handling
        self.Passing_Kicking = Passing_Kicking
        self.Dribbling_Reflexes = Dribbling_Reflexes
        self.Defending_Pace = Defending_Pace
        self.GK_Kicking = GK_Kicking
        self.GK_Positioning = GK_Positioning
        self.GK_Handling = GK_Handling
        self.GK_Reflexes = GK_Reflexes
        self.GK_Diving = GK_Diving
        self.foot = foot



    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Release clause('000)": [self.release_clause],
                "Best overall": [self.best_overall],
                "Potential": [self.potential],
                "Overall rating": [self.overall_rating],
                "Wage('000)": [self.wage],
                "Age": [self.age],
                "Finishing": [self.finishing],
                "Composure": [self.composure],
                "Total movement": [self.total_movement],
                "Sprint speed": [self.sprint_speed],
                "International reputation": [self.international_reputation],
                "Best position": [self.best_position],
                # Additional fields
                "Height": [self.Height],
                "Weight": [self.Weight],
                "Growth": [self.Growth],
                "Total attacking": [self.Total_attacking],
                "Crossing": [self.Crossing],
                "Heading accuracy": [self.Heading_accuracy],
                "Short passing": [self.Short_passing],
                "Volleys": [self.Volleys],
                "Total skill": [self.Total_skill],
                "Dribbling": [self.Dribbling],
                "Curve": [self.Curve],
                "FK Accuracy": [self.FK_Accuracy],
                "Long passing": [self.Long_passing],
                "Ball control": [self.Ball_control],
                "Acceleration": [self.Acceleration],
                "Agility": [self.Agility],
                "Reactions": [self.Reactions],
                "Balance": [self.Balance],
                "Total power": [self.Total_power],
                "Shot power": [self.Shot_power],
                "Jumping": [self.Jumping],
                "Stamina": [self.Stamina],
                "Strength": [self.Strength],
                "Long shots": [self.Long_shots],
                "Total mentality": [self.Total_mentality],
                "Aggression": [self.Aggression],
                "Interceptions": [self.Interceptions],
                "Att. Position": [self.Att_Position],
                "Vision": [self.Vision],
                "Penalties": [self.Penalties],
                "Total defending": [self.Total_defending],
                "Defensive awareness": [self.Defensive_awareness],
                "Standing tackle": [self.Standing_tackle],
                "Sliding tackle": [self.Sliding_tackle],
                "Total goalkeeping": [self.Total_goalkeeping],
                "Total stats": [self.Total_stats],
                "Base stats": [self.Base_stats],
                "Pace / Diving": [self.Pace_Diving],
                "Shooting / Handling": [self.Shooting_Handling],
                "Passing / Kicking": [self.Passing_Kicking],
                "Dribbling / Reflexes": [self.Dribbling_Reflexes],
                "Defending / Pace": [self.Defending_Pace],
                "GK Kicking": [self.GK_Kicking],
                "GK Positioning": [self.GK_Positioning],
                "GK Handling": [self.GK_Handling],
                "GK Reflexes": [self.GK_Reflexes],
                "GK Diving": [self.GK_Diving],
                "foot": [self.foot]



            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
