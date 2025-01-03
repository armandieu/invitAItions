
"""

@author:
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from shiny import App, render, ui, reactive
from shiny.types import ImgData
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import pickle
from shinyswatch import theme

# Load data
clients_df = pd.read_excel('data/clients.xlsx')
transactions_df = pd.read_excel('data/transactions.xlsx')
actions_df = pd.read_excel('data/actions.xlsx')
data_fEng = pd.read_csv('data/trx_clients_before_event.csv')

from helpers import helpers
final_df = helpers.feature_engin(data_fEng)

#Import trained models
with open('models/basic_regressor.pkl', 'rb') as file:
    model_r = pickle.load(file)
with open('models/classifier.pkl', 'rb') as file:
    model_c = pickle.load(file) 

def loadModel(model_name):
    with open(model_name, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

#Setting palette for plots
custom_palette = sns.color_palette('Set1', n_colors=4)  
sns.set_palette(custom_palette) 

type_options = list(actions_df.action_type_label.unique())
universe_options = list(actions_df.action_universe.unique())
channel_options = list(actions_df.action_channel.unique())
category_options = list(actions_df.action_category_label.unique())
action_subcategory_options = list(actions_df.action_subcategory_label.unique())
action_label_options = list(actions_df.action_label.unique())

# Classifier
classifier_columns_uncorr = ['mean_gross_amount_Men_products',
       'mean_gross_amount_Woman Shoes',
       'mean_gross_amount_Women Accessory', 'mean_gross_amount_Women Bags',
       'mean_gross_amount_Women Ready-to-Wear',
       'mean_quantity_Other',
       'mean_quantity_Women Accessory',
        'total_gross_amount',
        'mean_gross_amount', 'mean_quantity',
       'num_transactions', 'customer_life_months',
       'action_type_label','action_subcategory_label',
       'duration_days','action_universe','action_channel']
classifier_columns_client = ['mean_gross_amount_Men_products',
       'mean_gross_amount_Woman Shoes',
       'mean_gross_amount_Women Accessory', 'mean_gross_amount_Women Bags',
       'mean_gross_amount_Women Ready-to-Wear',
       'mean_quantity_Other',
       'mean_quantity_Women Accessory',
        'total_gross_amount',
        'mean_gross_amount', 'mean_quantity',
       'num_transactions', 'customer_life_months'
       ]
classifier_action_columns = ['action_type_label','action_subcategory_label',
       'duration_days','action_universe','action_channel']
df_input_c = final_df[classifier_columns_client]
# Regressor
reg_columns_uncorr = ['mean_gross_amount_Men_products',
       'mean_gross_amount_Woman Shoes',
       'mean_gross_amount_Women Accessory', 'mean_gross_amount_Women Bags',
       'mean_gross_amount_Women Ready-to-Wear',
       'mean_quantity_Other',
       'mean_quantity_Women Accessory',
        'total_gross_amount',
        'mean_gross_amount', 'mean_quantity',
       'num_transactions', 'customer_life_months','duration_days','action_universe','action_label','action_category_label'
       ,'action_type_label','action_subcategory_label'
       ]
reg_action_columns = ['duration_days','action_universe','action_category_label'
       ,'action_type_label','action_subcategory_label']
reg_columns_client = ['mean_gross_amount_Men_products',
       'mean_gross_amount_Woman Shoes',
       'mean_gross_amount_Women Accessory', 'mean_gross_amount_Women Bags',
       'mean_gross_amount_Women Ready-to-Wear',
       'mean_quantity_Other',
       'mean_quantity_Women Accessory',
        'total_gross_amount',
        'mean_gross_amount', 'mean_quantity',
       'num_transactions', 'customer_life_months'
       ]
df_input_r = final_df[reg_columns_client]
target = 'real_gain_event'

#Dashboard interface
app_ui = ui.page_fluid(
    
    ui.row(
        
        ui.column(2,
            ui.output_image("image", '80px', '80px'),
            style='display: flex; align-items: center; justify-content: center',
            ),
        ui.column(
            8,
            ui.h2("Event Guest List Generator"),
            ui.row(
                    ui.column(8,
                        ui.input_action_button('get_guest_list',"Create guess list ", class_='btn-primary'),
                        ui.input_action_button('btn',"Send Invitations ", class_='btn-primary')
                    ,
                    style= "text-align: right"
                    )
            )
        ),
        ui.column(2,
            ui.card(
                ui.card_header("Event Utility"),                
                ui.output_text_verbatim('utility', placeholder=True),
                ui.card_footer(""),
                full_screen=True,
            )
        )
        ),
    ui.layout_sidebar(
        ui.sidebar(
               ui.row(
                    ui.column(12,
                        ui.card(
                              ui.input_select("action_type_label", "Type:", choices=type_options, selected='Social Celebrity Action'),
                              ui.input_select("action_category_label", "Category:", choices=category_options,selected='Client'),
                              ui.input_select("action_subcategory_label", "Sub-Category:", choices=action_subcategory_options,selected='Lauch'),
                              ui.input_select("action_universe", "Universe:", choices=universe_options, selected="Men's Fashion"),
                              ui.input_select("channel", "Channel:", choices=channel_options, selected="In store"),
                              ui.input_select("action_label", "Action label:", choices=action_label_options, selected="Outdoor Event"),
                              ui.input_numeric("duration_days", "Duration:", 15),
                              ui.input_numeric("cost", "Cost (per person):", 150),
                              ui.input_numeric("n_people", "No. People:", 10),
                        )
                              ),
                ),
                width='400px'
            ),
                    ui.column(12,
                              
                              ui.row(
                                ui.card(
                                    ui.card_header("Guest List"),                                    
                                    ui.output_data_frame("summary_data"),
                                    full_screen=True,
                                )),
                              
                        ),
                    ),             
theme=theme.minty()
)

def server(input, output, session):
    
    @reactive.Effect
    @reactive.event(input.get_guest_list)
    def get_guest_list():
        # Access input values using input object and cast to float
        action_data = {
            'action_type_label': input.action_type_label(),
            'action_subcategory_label': input.action_subcategory_label(),
            'action_universe': input.action_universe(),
            'action_channel': input.channel(),
            'duration_days': input.duration_days(),
            'n_people': int(input.n_people()),
            'action_label': input.action_label(),
            'action_category_label':input.action_category_label(),
            'cost':float(input.cost()),
        }

        for column, value in action_data.items():
            final_df[column] = value
            
        df_input_r = final_df[reg_columns_uncorr]
        df_input_c = final_df[classifier_columns_uncorr]
        factor = 0.25
        factor_prob = 0.8
        skip = 3
        predicted_gain = model_r.predict(df_input_r)
        predicted_probs = model_c.predict_proba(df_input_c)
        # prediction = predict_credit(df_input, cluster_num[0])  # check columns and input
        df_result = pd.DataFrame({'client_id': final_df['client_id'].values, 'Probability of attending': np.round(predicted_probs[:,1], decimals=4), 'Expected gain': predicted_gain})
        df_result['Expected gain'] = df_result['Expected gain']*factor
        df_result['Probability of attending'] = df_result['Probability of attending']*factor_prob
        df_result['score'] = df_result['Probability of attending']*df_result['Expected gain']
        sorted_df = df_result.sort_values(by='score', ascending=False)
        
        sorted_df = sorted_df[skip:action_data['n_people']+skip]
        @output
        @render.data_frame
        def summary_data():
            # Round the 'score' and 'Expected gain' columns
            sorted_df['score'] = sorted_df['score'].round()
            sorted_df['Expected gain'] = sorted_df['Expected gain'].round()
            formatted_df = sorted_df.copy()  # Create a copy of the DataFrame
            formatted_df['Expected gain'] = formatted_df['Expected gain'].map('{:,.2f} \u20AC'.format)
            formatted_df['score'] = formatted_df['score'].map('{:,.2f} \u20AC'.format)
            formatted_df['Probability of attending'] = formatted_df['Probability of attending'].map('{:.2%}'.format)
            return render.DataGrid(
                formatted_df,
                row_selection_mode="multiple",
                width="100%",
                height="100%",
            )
        @output
        @render.text
        def utility():
            result_in_euros = sorted_df['score'].sum() - (action_data['n_people'] * action_data['cost'])
            return f"{result_in_euros:,.2f} \u20AC"  # Using \u20AC for the Euro sign

    @render.image
    def image():
        from pathlib import Path

        dir = Path(__file__).resolve().parent
        print(dir)
        img: ImgData = {"src": str(dir)+"/img/logo.jpg", "width": "100px", 'height': '100px'}
        return img
app = App(app_ui, server, debug=True)
      