import numpy as np
import pandas as pd
import re
import pickle

def get_time_date(date):
    time_stamp = pd.to_datetime(date)

    return pd.DataFrame({"Time": time_stamp.apply(lambda x: x.hour * 60 + x.minute),
                         "Weekday": time_stamp.apply(lambda x: x.weekday())})


def blocks_by_frequency(data):
    data['52BLOCK'] = (data['Block'] == "001XX N STATE ST")
    data['44BLOCK'] = (data['Block'] == "0000X W TERMINAL ST")
    data['31BLOCK'] = (data['Block'] == "0000X E ROOSEVELT RD")
    data['26BLOCKA'] = (data['Block'] == "100XX W OHARE ST")
    data['26BLOCKB'] = (data['Block'] == "011XX S CLARK ST")
    data['25BLOCKA'] = (data['Block'] == "064XX S DR MARTIN LUTHER KING JR DR")
    data['25BLOCKB'] = (data['Block'] == "083XX S STEWART AVE")
    data['23BLOCK'] = (data['Block'] == "008XX N MICHIGAN AVE")
    data.drop("Block", axis=1, inplace=True)
    return data


def location_desc(data):
    data['ISAPARTMENT'] = (data['Location Description'] == "APARTMENT")
    data['ISRES'] = (data['Location Description'] == "RESIDENCE")
    data['ISSTREET'] = (data['Location Description'] == "STREET")
    data['ISSIDEWALK'] = (data['Location Description'] == "SIDEWALK")
    data['ISRESIDENCE'] = (data['Location Description'] == "RESIDENCE - PORCH / HALLWAY")
    data['ISDEP'] = (data['Location Description'] == "DEPARTMENT STORE")
    data['ISALL'] = (data['Location Description'] == "ALLEY")
    data['ISREST'] = (data['Location Description'] == "RESTAURANT")
    data['ISCOM'] = (data['Location Description'] == "COMMERCIAL / BUSINESS OFFICE")
    data['ISGROC'] = (data['Location Description'] == "GROCERY")
    data['ISRES2'] = (data['Location Description'] == "RESIDENCE - YARD (FRONT / BACK)")
    data['ISVENIC'] = (data['Location Description'] == "VEHICLE NON-COMMERCIAL")
    data['ISGAS'] = (data['Location Description'] == "GAS STATION")
    return data


def drop_unnecessary(data):
    data.drop(["ID", "Case Number", "IUCR", "Description", "FBI Code", "Updated On", "Latitude", "Longitude",
               "Location", "Year", "Date"], axis=1, inplace=True)


def handle_description(data):
    unique_descriptions = pd.unique(data["Location Description"])
    num_range = np.arange(len(unique_descriptions))
    x = pd.Series(num_range, index=unique_descriptions).to_dict()
    return data.replace(x)


def handle_xcoordinate(data):
    nulls = data[data['X Coordinate'].isnull()]
    not_nulls = data[data['X Coordinate'].isnull() == False]
    indexes = data.index[data['X Coordinate'].isnull()].tolist()
    indexCounter = 0
    for i in range(len(nulls)):
        district, ward, beat, community_area = nulls['District'].iloc[i], \
                                               nulls['Ward'].iloc[i], \
                                               nulls['Beat'].iloc[i], nulls["Community Area"].iloc[i]

        matcher = not_nulls.loc[(not_nulls['District'] == district) & (not_nulls['Ward'] == ward)
                                & (not_nulls['Beat'] == beat) & (not_nulls['Community Area'] == community_area)]
        if len(matcher) != 0:
            data.at[indexes[indexCounter], "X Coordinate"] = matcher['X Coordinate'].mean()
            indexCounter += 1
            continue
        sum = 0
        counter = 0
        matcher = not_nulls.loc[(not_nulls['District'] == district)]
        if len(matcher) != 0:
            counter += 1
            sum += matcher['X Coordinate'].mean()
        matcher = not_nulls.loc[(not_nulls['Ward'] == ward)]
        if len(matcher) != 0:
            counter += 1
            sum += matcher['X Coordinate'].mean()
        matcher = not_nulls.loc[(not_nulls['Community Area'] == community_area)]
        if len(matcher) != 0:
            counter += 1
            sum += matcher['X Coordinate'].mean()
        matcher = not_nulls.loc[(not_nulls['Beat'] == beat)]
        if len(matcher) != 0:
            counter += 1
            sum += matcher['X Coordinate'].mean()
        if counter != 0:
            data.at[indexes[indexCounter], "X Coordinate"] = sum / counter
            indexCounter += 1
            continue
        data.at[indexes[indexCounter], "X Coordinate"] = data['X Coordinate'].mean()
        indexCounter += 1


def handle_ycoordinate(data):
    nulls = data[data['Y Coordinate'].isnull()]
    not_nulls = data[data['Y Coordinate'].isnull() == False]
    indexes = data.index[data['Y Coordinate'].isnull()].tolist()
    indexCounter = 0
    for i in range(len(nulls)):
        district, ward, beat, community_area = nulls['District'].iloc[i], \
                                               nulls['Ward'].iloc[i], \
                                               nulls['Beat'].iloc[i], nulls["Community Area"].iloc[i]

        matcher = not_nulls.loc[(not_nulls['District'] == district) & (not_nulls['Ward'] == ward)
                                & (not_nulls['Beat'] == beat) & (not_nulls['Community Area'] == community_area)]
        if len(matcher) != 0:
            data.at[indexes[indexCounter], "Y Coordinate"] = matcher['Y Coordinate'].mean()
            indexCounter += 1
            continue
        sum = 0
        counter = 0
        matcher = not_nulls.loc[(not_nulls['District'] == district)]
        if len(matcher) != 0:
            counter += 1
            sum += matcher['Y Coordinate'].mean()
        matcher = not_nulls.loc[(not_nulls['Ward'] == ward)]
        if len(matcher) != 0:
            counter += 1
            sum += matcher['Y Coordinate'].mean()
        matcher = not_nulls.loc[(not_nulls['Community Area'] == community_area)]
        if len(matcher) != 0:
            counter += 1
            sum += matcher['Y Coordinate'].mean()
        matcher = not_nulls.loc[(not_nulls['Beat'] == beat)]
        if len(matcher) != 0:
            counter += 1
            sum += matcher['Y Coordinate'].mean()
        if counter != 0:
            data.at[indexes[indexCounter], "Y Coordinate"] = sum / counter
            indexCounter += 1
            continue
        data.at[indexes[indexCounter], "Y Coordinate"] = data['Y Coordinate'].mean()
        indexCounter += 1


def preprocess_data(data, pred=False):
    arrest_mode = data.mode()['Arrest'][0]
    domestic_mode = data.mode()['Domestic'][0]
    DATE_REGEX = re.compile('[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]')
    mean_year = int(data["Year"].mean())
    template_date = "03/01/" + str(mean_year)
    map_response_rev = {"BATTERY": 0, "THEFT": 1, "CRIMINAL DAMAGE": 2,
                        "DECEPTIVE PRACTICE": 3, "ASSAULT": 4}
    map_response = {0: "BATTERY", 1: "THEFT", 2: "CRIMINAL DAMAGE",
                    3: "DECEPTIVE PRACTICE", 4: "ASSAULT"}

    def handle_empty_date(row):
        if pd.isna(row):
            return template_date
        else:
            f_match = re.match(DATE_REGEX, row)
            if f_match is None:
                return template_date
        return row

    def handle_empty_loc_desc(row):
        if pd.isna(row):
            return "OTHER (SPECIFY)"
        return row

    def handle_arrest(row):
        if pd.isna(row):
            return arrest_mode
        return row

    def handle_domestic(row):
        if pd.isna(row):
            return domestic_mode
        return row

    def handle_locations(row):
        if pd.isna(row):
            return '-1'
        return row

    def apply_primary(row):
        return map_response_rev[row]

    # date
    data['Date'] = data['Date'].apply(handle_empty_date)
    # location descreption
    data['Location Description'] = data['Location Description'].apply(handle_empty_loc_desc)
    # Arrest
    data['Arrest'] = data['Arrest'].apply(handle_arrest)
    data['Domestic'] = data['Domestic'].apply(handle_domestic)
    data['District'] = data['District'].apply(handle_locations)
    data['Ward'] = data['Ward'].apply(handle_locations)
    data['Beat'] = data['Beat'].apply(handle_locations)
    data['Community Area'] = data['Community Area'].apply(handle_locations)

    handle_xcoordinate(data)
    handle_ycoordinate(data)

    data = pd.concat([data, get_time_date(data["Date"])], axis=1)
    drop_unnecessary(data)
    data = blocks_by_frequency(data)
    data = handle_description(data)
    data = location_desc(data)

    if pred:
        return data.to_numpy()

    # primary type
    data = data.loc[(data['Primary Type'] == "THEFT") | (data['Primary Type'] == "ASSAULT")
                    | (data['Primary Type'] == "BATTERY") | (data['Primary Type'] == "CRIMINAL DAMAGE")
                    | (data['Primary Type'] == "DECEPTIVE PRACTICE")]
    data['Primary Type'] = data['Primary Type'].apply(apply_primary)

    Y = data["Primary Type"]

    X = data.drop("Primary Type", axis=1)

    return X.to_numpy(), Y.to_numpy()


def save_model(model):
    with open("model.sav", "wb") as model_fd:
        pickle.dump(model, model_fd)


def load_model():
    with open("model.sav", "rb") as model_fd:
        m = pickle.load(model_fd)
    return m