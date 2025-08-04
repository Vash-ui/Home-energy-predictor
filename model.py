import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def preprocess_data(df):
    df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)
    df['season'] = df['date'].dt.month % 12 // 3 + 1

    bins = [0, 6, 12, 17, 22, 24]
    labels = ['Late Night', 'Morning', 'Afternoon', 'Evening', 'Night']
    df['hour_bin'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)

    features = ['T2', 'T6', 'RH_1', 'RH_2', 'T_out', 'RH_out', 'Windspeed',
                'hour', 'day', 'weekday', 'is_weekend', 'season']

    df = df.dropna(subset=features + ['Appliances'])

    X = df[features]
    X = pd.get_dummies(X, columns=['season'], drop_first=True)
    y = df['Appliances']

    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns.tolist()

def train():
    # Load your dataset here
    df = pd.read_csv("C:\\Users\\SIDHARTH H\\OneDrive\\Desktop\\energy_predictor\\data\\energydata_complete.csv")

    (X_train, X_test, y_train, y_test), feature_columns = preprocess_data(df)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    mse_lr = mean_squared_error(y_test, lr.predict(X_test))
    mse_rf = mean_squared_error(y_test, rf.predict(X_test))

    return lr, rf, mse_lr, mse_rf, feature_columns
