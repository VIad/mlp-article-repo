import pandas as pd

def load_train_data():
    df = pd.read_csv("train_no_names.csv")
    df = df.drop(columns=["what", "Cabin", "PassengerId", "Ticket"], axis=1)
    df['Sex'] = df['Sex'].replace(['male'], "0")
    df['Sex'] = df['Sex'].replace(['female'], "1")
    df['Embarked'] = df['Embarked'].replace(['S'], "0") # Should be one-hot
    df['Embarked'] = df['Embarked'].replace(['C'], "0.2")
    df['Embarked'] = df['Embarked'].replace(['Q'], "0.5")
    df['Embarked'] = df['Embarked'].astype(float)
    df['Sex'] = df['Sex'].astype(float)
    df['Pclass'] = df['Pclass'].astype(float)

    print("MAX / MIN AGE")
    print(df['Age'].max(), df['Age'].min(), )
    print("MAX / MIN SibSp")
    print(df['SibSp'].max(), df['SibSp'].min(), )
    print("MAX / MIN Parch")
    print(df['Parch'].max(), df['Parch'].min(), )
    print("MAX / MIN Fare")
    print(df['Fare'].max(), df['Fare'].min(), )

    df = df.dropna()

    print(df.isnull().sum())

    np_arr = df.to_numpy()

    return np_arr