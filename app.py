import numpy as np
import pandas as pd
import gradio as gr
import lightgbm as lgb

model = lgb.Booster(model_file='lgbm_model-2.txt')
df_test = pd.read_pickle('df_test-2.pkl')

def predict(bid_size:int = 50_000, 
            ask_size:int = 50_000, 
            bid_price:float = 0.99, 
            ask_price: float = 1.01, 
            imbalance_buy_sell_flag:int = 1, 
            imbalance_size:int = 5_000_000, 
            matched_size:int = 100_000_000, 
            reference_price:float = 0.9955,
            target_id:int = 0):
    wap = (float(bid_price) * float(ask_size) + float(ask_price) * float(bid_size)) / (float(bid_size) + float(ask_size))
    prediction_array = np.array([bid_size, 
                                 ask_size, 
                                 bid_price, 
                                 ask_price, 
                                 wap,
                                 imbalance_buy_sell_flag, 
                                 imbalance_size, 
                                 matched_size, 
                                 reference_price]).reshape(1, -1)
    return model.predict(prediction_array)[0]

def generate_features():
    i = np.random.randint(2000)
    bid_size = df_test['bid_size'].iloc[i:i+1].values[0]
    ask_size = df_test['ask_size'].iloc[i:i+1].values[0]
    bid_price = df_test['bid_price'].iloc[i:i+1].values[0]
    ask_price = df_test['ask_price'].iloc[i:i+1].values[0]
    imbalance_buy_sell_flag = df_test['imbalance_buy_sell_flag'].iloc[i:i+1].values[0]
    imbalance_size = df_test['imbalance_size'].iloc[i:i+1].values[0]
    matched_size = df_test['matched_size'].iloc[i:i+1].values[0]
    reference_price = df_test['reference_price'].iloc[i:i+1].values[0] 
    target = df_test['target'].iloc[i:i+1].values[0] 
    return bid_size, ask_size, bid_price, ask_price, imbalance_buy_sell_flag, imbalance_size, matched_size, reference_price, target

with gr.Blocks() as demo:
    gen_button = gr.Button(value = 'Generate features')

    with gr.Row() as row1:
        bid_size = gr.Textbox(label="Bid size")
        bid_price = gr.Textbox(label="Bid price")
        ask_size = gr.Textbox(label="Ask size")
        ask_price = gr.Textbox(label="Ask price")

    with gr.Row() as row2:
        imbalance_buy_sell_flag = gr.Textbox(label="Imbalance on")
        imbalance_size = gr.Textbox(label="Imbalance size")
        matched_size = gr.Textbox(label="Matched size")
        reference_price = gr.Textbox(label="Reference price")
    
    submit = gr.Button(value = 'Predict')
    
    output = gr.Textbox(label = "The weighted average price in 1 minute is:", interactive = False,)
    target = gr.Textbox(label = "The true target is:", interactive = False,)

    gen_button.click(
        fn=generate_features,
        outputs=[bid_size, ask_size, bid_price, ask_price, imbalance_buy_sell_flag, imbalance_size, matched_size, reference_price, target]
    )    
    
    submit.click(predict, 
                 inputs = [bid_size, ask_size, bid_price, ask_price, imbalance_buy_sell_flag, imbalance_size, matched_size, reference_price], 
                 outputs = [output])    

demo.launch(share = False, debug = False)