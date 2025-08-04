import pandas as pd

def prepare_input(input_dict):
    """
    Convert user inputs (dict) into model-ready dataframe row
    """
    df = pd.DataFrame([input_dict])

    # One-hot encode season feature to match training
    season = df.loc[0, 'season']
    for s in [2, 3, 4]:  # seasons 2,3,4 after drop_first=True
        col = f'season_{s}'
        df[col] = 1 if season == s else 0

    # Drop original season column
    df = df.drop(columns=['season'])

    return df
