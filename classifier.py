import Preprocessing
import Clustering
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def send_police_cars(date_list):
    ret_list = []
    day_cluster_dic = Clustering.load_j_cluster()
    for date in date_list:

        index = np.random.choice(len(day_cluster_dic["0"]), 30, replace=False)

        temp_date = pd.to_datetime(date)
        pred_day = temp_date.weekday()
        points = np.array(day_cluster_dic[str(pred_day)])[index]

        t = np.apply_along_axis(
                lambda x: pd.Timestamp(year=temp_date.year, month=temp_date.month,
                                            day=temp_date.day, hour=int(x[0] // 60),
                                            minute=int(x[0] % 60)),
                axis=1, arr=points[:, -1].reshape((-1, 1)))

        temp_list = []
        for i in range(len(t)):
            temp_list.append((points[i][0], points[i][1], t[i]))

        ret_list.append(temp_list)

    return ret_list


def predict(path):
    X_pred = pd.read_csv(path, index_col=0)

    random_forest_model = Preprocessing.load_model()

    X_ = Preprocessing.preprocess_data(X_pred, pred=True)

    return random_forest_model.predict(X_)


if __name__ == '__main__':
    df = pd.read_csv("Dataset_crimes.csv", index_col=0)

    X, y = Preprocessing.preprocess_data(df)

    # train model
    random_forest = RandomForestClassifier(n_estimators=200, max_depth=7)

    random_forest.fit(X, y)

    Preprocessing.save_model(random_forest)

    # create clustering
    points_d = Clustering.points_per_day(df)














