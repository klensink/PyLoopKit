import json
import datetime
from datetime import time, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.offline import plot
import pytz
import os

from pyloopkit.dose import DoseType
from pyloopkit.generate_graphs import plot_graph, plot_loop_inspired_glucose_graph
from pyloopkit.loop_math import predict_glucose
from pyloopkit.loop_data_manager import update
from pyloopkit.pyloop_parser import (
    parse_report_and_run, parse_dictionary_from_previous_run
)

import matplotlib.pyplot as plt
import simglucose
from simglucose.simulation.user_interface import simulate
from simglucose.controller.base import Controller, Action

from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim

def parse_json(path, name):
    """ Get a dictionary output from a previous run of PyLoopKit
        and convert the ISO strings to datetime or time objects, and
        dose types to enums
    """
    data_path_and_name = os.path.join(path, name)

    with open(data_path_and_name, "r") as file:
        dictionary = json.load(file)

    keys_with_times = [
        "basal_rate_start_times",
        "carb_ratio_start_times",
        "sensitivity_ratio_start_times",
        "sensitivity_ratio_end_times",
        "target_range_start_times",
        "target_range_end_times"
        ]

    for key in keys_with_times:
        new_list = []
        for string in dictionary.get(key):
            new_list.append(time.fromisoformat(string))
        dictionary[key] = new_list

    keys_with_datetimes = [
        "dose_start_times",
        "dose_end_times",
        "glucose_dates",
        "carb_dates"
        ]

    for key in keys_with_datetimes:
        new_list = []
        for string in dictionary.get(key):
            new_list.append(datetime.datetime.fromisoformat(string))
        dictionary[key] = new_list

    dictionary["time_to_calculate_at"] = datetime.datetime.fromisoformat(
        dictionary["time_to_calculate_at"]
    )

    last_temp = dictionary.get("last_temporary_basal")
    dictionary["last_temporary_basal"] = [
        DoseType.from_str(last_temp[0]),
        datetime.datetime.fromisoformat(last_temp[1]),
        datetime.datetime.fromisoformat(last_temp[2]),
        last_temp[3]
    ]

    dictionary["dose_types"] = [
        DoseType.from_str(value) for value in dictionary.get("dose_types")
    ]

    return dictionary

# save dictionary as json file
def convert_times_and_types(obj):
    """ Convert dates and dose types into strings when saving as a json """
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.time):
        return obj.isoformat()
    if isinstance(obj, DoseType):
        return str(obj.name)

def strip_tz(date):
    return date.replace(tzinfo=None)

def str_to_time(s, fmt='%H:%M:%S'):
    return datetime.datetime.strptime(s, fmt).time()

def parse_input_string(s):
    try:
        return str_to_time(s)
    except:
        return s

def parse_json_settings(d):
    """
    Convert all '%H:%M:%S values in a json to datetime.time objects. 
    Looks recusively through dictionarys and lists.
    """

    for k,v in d.items():
        if isinstance(v, str):
            d[k] = parse_input_string(v)
        elif isinstance(v, list):
            d[k] = list(map(
                parse_input_string,
                v
            ))
        elif isinstance(v, dict):
            d[k] = parse_json_settings(v)
    
    return d

def make_loop_plot(recommendations):

    inputs = recommendations.get('input_data')
    current_time = inputs.get("time_to_calculate_at")

    # blood glucose data
    glucose_dates = pd.DataFrame(inputs.get("glucose_dates"), columns=["time"])
    glucose_values = pd.DataFrame(inputs.get("glucose_values"), columns=["mg_dL"])
    bg = pd.concat([glucose_dates, glucose_values], axis=1)

    # Set bg color values
    bg['bg_colors'] = 'mediumaquamarine'
    bg.loc[bg['mg_dL'] < 54, 'bg_colors'] = 'indianred'
    low_location = (bg['mg_dL'] > 54) & (bg['mg_dL'] < 70)
    bg.loc[low_location, 'bg_colors'] = 'lightcoral'
    high_location = (bg['mg_dL'] > 180) & (bg['mg_dL'] <= 250)
    bg.loc[high_location, 'bg_colors'] = 'mediumpurple'
    bg.loc[(bg['mg_dL'] > 250), 'bg_colors'] = 'slateblue'

    bg_trace = go.Scattergl(
        name="bg",
        x=bg["time"],
        y=bg["mg_dL"],
        hoverinfo="y+name",
        mode='markers',
        marker=dict(
            size=6,
            line=dict(width=0),
            color=bg["bg_colors"]
        )
    )

    # bolus data
    dose_start_times = (
        pd.DataFrame(inputs.get("dose_start_times"), columns=["startTime"])
    )
    dose_end_times = (
        pd.DataFrame(inputs.get("dose_end_times"), columns=["endTime"])
    )
    dose_values = (
        pd.DataFrame(inputs.get("dose_values"), columns=["dose"])
    )
    dose_types = (
        pd.DataFrame(inputs.get("dose_types"), columns=["type"])
    )

    dose_types["type"] = dose_types["type"].apply(convert_times_and_types)

    dose = pd.concat(
        [dose_start_times, dose_end_times, dose_values, dose_types],
        axis=1
    )

    unique_dose_types = dose["type"].unique()

    # bolus data
    if "bolus" in unique_dose_types:
        bolus = dose[dose["type"] == "bolus"]
        bolus_trace = go.Bar(
            name="bolus",
            x=bolus["startTime"],
            y=bolus["dose"],
            hoverinfo="y+name",
            width=999999,
            marker=dict(color='lightskyblue')
        )

    # basals rates
    # scheduled basal rate
    basal_rate_start_times = (
        pd.DataFrame(inputs.get("basal_rate_start_times"), columns=["time"])
    )
    basal_rate_minutes = (
        pd.DataFrame(inputs.get("basal_rate_minutes"), columns=["duration"])
    )
    basal_rate_values = (
        pd.DataFrame(inputs.get("basal_rate_values"), columns=["sbr"])
    )
    sbr = pd.concat(
        [basal_rate_start_times, basal_rate_minutes, basal_rate_values],
        axis=1
    )

    # create a contiguous basal time series
    bg_range = pd.date_range(
        bg["time"].min() - datetime.timedelta(days=1),
        current_time,
        freq="1s"
    )
    contig_ts = pd.DataFrame(bg_range, columns=["datetime"])
    contig_ts["time"] = contig_ts["datetime"].dt.time
    basal = pd.merge(contig_ts, sbr, on="time", how="left")
    basal["sbr"].fillna(method='ffill', inplace=True)
    basal.dropna(subset=['sbr'], inplace=True)

    # temp basal data
    if ("basal" in unique_dose_types) | ("suspend" in unique_dose_types):
        temp_basal = (
            dose[((dose["type"] == "basal") | (dose["type"] == "suspend"))]
        )

        temp_basal["type"].replace("basal", "temp", inplace=True)
        all_temps = pd.DataFrame()
        for idx in temp_basal.index:
            rng = pd.date_range(
                temp_basal.loc[idx, "startTime"],
                temp_basal.loc[idx, "endTime"] - datetime.timedelta(seconds=1),
                freq="1s"
            )
            temp_ts = pd.DataFrame(rng, columns=["datetime"])
            temp_ts["tbr"] = temp_basal.loc[idx, "dose"]
            temp_ts["type"] = temp_basal.loc[idx, "type"]
            all_temps = pd.concat([all_temps, temp_ts])

        basal = pd.merge(basal, all_temps, on="datetime", how="left")
        basal["type"].fillna("scheduled", inplace=True)

    else:
        basal["tbr"] = np.nan

    basal["delivered"] = basal["tbr"]
    basal.loc[basal["delivered"].isnull(), "delivered"] = (
        basal.loc[basal["delivered"].isnull(), "sbr"]
    )

    sbr_trace = go.Scatter(
        name="scheduled",
        mode='lines',
        x=basal["datetime"],
        y=basal["sbr"],
        hoverinfo="y+name",
        showlegend=False,
        line=dict(
            shape='vh',
            color='cornflowerblue',
            dash='dot'
        )
    )

    basal_trace = go.Scatter(
        name="delivered",
        mode='lines',
        x=basal["datetime"],
        y=basal["delivered"],
        hoverinfo="y+name",
        showlegend=False,
        line=dict(
            shape='vh',
            color='cornflowerblue'
        ),
        fill='tonexty'
    )

    # carb data
    # carb-to-insulin-ratio
    carb_ratio_start_times = (
        pd.DataFrame(inputs.get("carb_ratio_start_times"), columns=["time"])
    )
    carb_ratio_values = (
        pd.DataFrame(inputs.get("carb_ratio_values"), columns=["cir"])
    )
    cir = pd.concat([carb_ratio_start_times, carb_ratio_values], axis=1)

    carbs = pd.merge(contig_ts, cir, on="time", how="left")
    carbs["cir"].fillna(method='ffill', inplace=True)
    carbs.dropna(subset=['cir'], inplace=True)

    # carb events
    carb_dates = pd.DataFrame(inputs.get("carb_dates"), columns=["datetime"])
    carb_values = pd.DataFrame(inputs.get("carb_values"), columns=["grams"])
    carb_absorption_times = (
        pd.DataFrame(
            inputs.get("carb_absorption_times"),
            columns=["aborption_time"]
        )
    )
    carb_events = (
        pd.concat([carb_dates, carb_values, carb_absorption_times], axis=1)
    )

    carbs = pd.merge(carbs, carb_events, on="datetime", how="left")

    # add bolus height for figure
    carbs["bolus_height"] = carbs["grams"] / carbs["cir"]

    carb_trace = go.Scatter(
        name="carbs",
        mode='markers + text',
        x=carbs["datetime"],
        y=carbs["bolus_height"] + 2,
        hoverinfo="name",
        marker=dict(
            color='gold',
            size=25
        ),
        showlegend=False,
        text=carbs["grams"],
        textposition='middle center'
    )

    # combine the plots
    basal_trace.yaxis = "y"
    sbr_trace.yaxis = "y"
    bolus_trace.yaxis = "y2"
    carb_trace.yaxis = "y2"
    bg_trace.yaxis = "y3"

    data = [basal_trace, sbr_trace, bolus_trace, carb_trace, bg_trace]
    layout = go.Layout(
        yaxis=dict(
            domain=[0, 0.2],
            range=[0, max(basal["sbr"].max(), basal["tbr"].max()) + 1],
            fixedrange=True,
            hoverformat=".2f",
            title=dict(
                text="Basal Rate U/hr",
                font=dict(
                    size=12
                )
            )
        ),
        showlegend=False,
        yaxis2=dict(
            domain=[0.25, 0.45],
            range=[0, max(bolus["dose"].max(), carbs["bolus_height"].max()) + 10],
            fixedrange=True,
            hoverformat=".1f",
            title=dict(
                text="Bolus U",
                font=dict(
                    size=12
                )
            )
        ),
        yaxis3=dict(
            domain=[0.5, 1],
            range=[0, 402],
            fixedrange=True,
            hoverformat=".0f",
            title=dict(
                text="Blood Glucose mg/dL",
                font=dict(
                    size=12
                )
            )
        ),
        xaxis=dict(
            range=(
                current_time - datetime.timedelta(days=1),
                current_time + datetime.timedelta(minutes=60)
            )
        ),
        dragmode="pan",
    )

    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='test'.split(".")[0] + '-output.html')


def plot_recommendations(recommendations):

    inputs = recommendations.get('input_data')

    start_dates = []
    end_dates = []
    if inputs.get('glucose_dates'):
        start_dates.append(inputs.get('glucose_dates')[0])
        end_dates.append(recommendations.get('predicted_glucose_dates')[-1])
    if inputs.get('carb_dates'):
        start_dates.append(inputs.get('carb_dates')[0])
        end_dates.append(inputs.get('carb_dates')[-1])
    if inputs.get('dose_start_times'):
        start_dates.append(inputs.get('dose_start_times')[0])
        end_dates.append(inputs.get('dose_start_times')[-1])
    
    start_date = strip_tz(min(start_dates))
    end_date = strip_tz(max(end_dates))

    # Plot the predicted Glucose
    fig, (ax_glucose, ax_insulin, ax_carbs) = plt.subplots(3, 1, figsize=(12,8))


    if inputs.get('glucose_dates'):
        ax_glucose.plot(
            list(map(strip_tz, recommendations['input_data']['glucose_dates'])),
            recommendations['input_data']['glucose_values'],
            'bo'
        )
        ax_glucose.plot(
            list(map(strip_tz, recommendations['predicted_glucose_dates'])),
            recommendations['predicted_glucose_values'],
            'b--'
        )
    ax_glucose.set_xlim([start_date, end_date])
    ax_glucose.set_ylim([0, 400])

    if inputs.get('dose_start_times'):
        ax_insulin.plot(
            list(map(strip_tz, recommendations['input_data']['dose_start_times'])),
            recommendations['input_data']['dose_values'],
            'rv'
        )
    ax_insulin.set_xlim([start_date, end_date])

    if inputs.get('carb_dates'):
        ax_carbs.plot(
            list(map(strip_tz, recommendations['input_data']['carb_dates'])),
            recommendations['input_data']['carb_values'],
            'gs'
        )
    ax_carbs.set_xlim([start_date, end_date])

    plt.show()

def clear_history(inputs):
    # Add glucose events
    # inputs['glucose_dates'] = inputs['glucose_dates'][-19:-2]
    # inputs['glucose_values'] = inputs['glucose_values'][-19:-2]
    now_date = datetime.datetime.now()
    inputs['glucose_dates'] = [
        # now_date - timedelta(minutes=15),
        # now_date - timedelta(minutes=10),
        # now_date - timedelta(minutes=5),
        # now_date
    ]
    inputs['glucose_values'] = [
        # 100.0,
        # 105.0,
        # 110.0,
        # 115.0
    ]

    # Clear out old carb events
    inputs['carb_dates'] = []
    inputs['carb_values'] = []
    inputs['carb_absorption_times'] = []

    # Clear out old insulin events 
    inputs['dose_start_times'] = []
    inputs['dose_end_times'] = []
    inputs['dose_values'] = []
    inputs['dose_types'] = []
    inputs['dose_delivered_units'] = []

    inputs['last_temporary_basal'] = []
    inputs['time_to_calculate_at'] = now_date

    return inputs

class LoopController(Controller):

    def __init__(self, start_date):
        # Load a previous run and trim out data
        name = "example_from_previous_run.json"
        path = "pyloopkit/example_files/"
        inputs = parse_json(path, name)
        self.inputs = clear_history(inputs)
        self.start_date = start_date
        self.now = start_date
        self.recommendations = None

    def policy(self, observation, reward, done, **info):
        sample_time = info.get('sample_time')
        self.now = self.now + timedelta(minutes=sample_time)
        self.inputs['time_to_calculate_at'] = self.now

        # Update CGM data
        self.inputs['glucose_dates'].append(self.now)
        self.inputs['glucose_values'].append(observation.CGM)

        # Update carbs
        meal = info.get('meal')
        if meal > 0:
            self.inputs['carb_dates'].append(self.now)
            self.inputs['carb_values'].append(meal)
            self.inputs['carb_absorption_times'].append(180)

        # Get Scheduled Basal rate
        

        # Run Loop and get recommendations
        try:
            recommendations = update(self.inputs)
        except:
            print('Red Loop', self.now)
            recommendations = None
        self.recommendations = recommendations

        if recommendations:

            ### Meal Bolus
            if meal > 0:

                # Give a bolus
                bolus = recommendations.get('recommended_bolus')
                bolus_rate = bolus[0]/sample_time

                # Record the bolus
                self.inputs['dose_start_times'].append(self.now)
                self.inputs['dose_end_times'].append(self.now + timedelta(minutes=1))
                self.inputs['dose_values'].append(bolus[0])
                self.inputs['dose_types'].append(DoseType.from_str('Bolus'))
                self.inputs['dose_delivered_units'].append(bolus[0])

            else:
                bolus_rate = 0

            ### Temp Basals
            if recommendations.get('recommended_temp_basal'):
                (temp_basal_rate, temp_basal_duration) = recommendations.get('recommended_temp_basal')

                # Use the new temp basal rate in the pump action
                temp_basal_rate_min = temp_basal_rate/60

                # Record the dose into input data
                self.inputs['dose_start_times'].append(self.now)
                self.inputs['dose_end_times'].append(self.now + timedelta(minutes=temp_basal_duration))
                self.inputs['dose_values'].append(temp_basal_rate)
                self.inputs['dose_types'].append(DoseType.from_str('TempBasal'))
                self.inputs['dose_delivered_units'].append(temp_basal_rate_min * temp_basal_duration)
                print(temp_basal_rate)

                # Double check the last temp_basal to see if we need to change its end time
                last_temp_basal = self.inputs.get('last_temporary_basal')
                if last_temp_basal:
                    (dose_type, start_time, end_time, rate) = last_temp_basal
                    assert dose_type == DoseType.tempbasal or dose_type == DoseType.basal, "Last basal is an invalid dosetype"

                    # If the temp basal is still running, find it and record its end
                    if self.now < end_time:
                        ind = [i for i, s in enumerate(self.inputs.get('dose_start_times')) if s == start_time][-1]
                        self.inputs['dose_end_times'][ind] = self.now

                        duration = (self.now - start_time).seconds/3600
                        self.inputs['dose_delivered_units'][ind] = rate * duration

                assert len(self.inputs.get('dose_start_times')) == len(self.inputs.get('dose_end_times'))

                # Update last temp basal
                self.inputs['last_temporary_basal'] = [
                    DoseType.from_str('TempBasal'),
                    self.now,
                    self.now + timedelta(minutes=temp_basal_duration),
                    temp_basal_rate
                ]
            else: 
                temp_basal_rate_min = self.inputs.get('basal_rate_values')[0]/60

            action = Action(basal=temp_basal_rate_min, bolus=bolus_rate)

        else:
            action = Action(basal = self.inputs.get('basal_rate_values')[0]/60, bolus = 0)

        print(self.now, observation, action.basal*60, action.bolus*sample_time)
        return action

    def reset(self):
        raise NotImplementedError()
        self.inputs = clear_history(self.inputs)

if __name__ == "__main__":


    ## Setup Sim Env
    # specify start_time as the beginning of today
    now = datetime.datetime.now()
    start_time = datetime.datetime.combine(now.date(), datetime.datetime.min.time())

    # --------- Create Random Scenario --------------
    # Specify results saving path
    path = './results'

    # Create a simulation environment
    patient = T1DPatient.withName('adolescent#001')
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')

    # custom scenario is a list of tuples (time, meal_size)
    scen = [
        # (1, 12055), 
        (0, 1), 
    ]
    scenario = CustomScenario(start_time=start_time, scenario=scen)
    env = T1DSimEnv(patient, sensor, pump, scenario)

    # Create a controller
    controller = LoopController(start_time)
    # controller = BBController()

    # Put them together to create a simulation object
    s1 = SimObj(env, controller, timedelta(hours=6), animate=False, path=path)
    results1 = sim(s1)
    plt.plot(results1.CGM)
    plt.ylim([0, 300])
    plt.grid()
    plt.show()

    make_loop_plot(controller.recommendations)
    print(results1)
