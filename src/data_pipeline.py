import pandas as pd
import src.util as utils
import copy
from sklearn.model_selection import train_test_split

def read_raw_data(config: dict) -> pd.DataFrame:
    # Return raw dataset
    return pd.read_csv(config["dataset_path"])

def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
    input_data = copy.deepcopy(input_data)
    config = copy.deepcopy(config)

    if not api:
        # Check range of outcome
        assert input_data[config["int_columns"][6]].between(
                config["range_fire_alarm"][0],
                config["range_fire_alarm"][1]
                ).sum() == len(input_data), "an error occurs in range_outcome."

    else:
        # In case checking data from api
        # Last column names in list of int columns are not used as predictor (Outcome)
        int_columns = config["int_columns"]
        del int_columns[-1:]

        # All column names in list of float columns
        float_columns = config["float_columns"]
 
    # Check range of age
    assert input_data[config["int_columns"][0]].between(
            config["range_age"][0],
            config["range_age"][1]
            ).sum() == len(input_data), "an error occurs in range_age."
    
    # Check range of pregnancies
    assert input_data[config["int_columns"][1]].between(
            config["range_pregnancies"][0],
            config["range_pregnancies"][1]
            ).sum() == len(input_data), "an error occurs in range_pregnancies."
    
    # Check range of glucose
    assert input_data[config["int_columns"][2]].between(
            config["range_glucose"][0],
            config["range_glucose"][1]
            ).sum() == len(input_data), "an error occurs in range_glucose."
    
    # Check range of blood pressure
    assert input_data[config["int_columns"][3]].between(
            config["range_blood_pressure"][0],
            config["range_blood_pressure"][1]
            ).sum() == len(input_data), "an error occurs in range_blood_pressure."
 
    # Check range of skin thickness
    assert input_data[config["int_columns"][4]].between(
            config["range_skin_thickness"][0],
            config["range_skin_thickness"][1]
            ).sum() == len(input_data), "an error occurs in range_skin_thickness."
    
    # Check range of insulin
    assert input_data[config["int_columns"][5]].between(
            config["range_insulin"][0],
            config["range_insulin"][1]
            ).sum() == len(input_data), "an error occurs in range_insulin."
    
    # Check range of bmi
    assert input_data[config["float_columns"][0]].between(
            config["range_bmi"][0],
            config["range_bmi"][1]
            ).sum() == len(input_data), "an error occurs in range_bmi."
    
    # Check range of diabetes pedigree function
    assert input_data[config["float_columns"][1]].between(
            config["range_diabetes_pedigree_function"][0],
            config["range_diabetes_pedigree_function"][1]
            ).sum() == len(input_data), "an error occurs in range_diabetes_pedigree_function."

def split_data(input_data: pd.DataFrame, config: dict):
    # Split predictor and label
    x = input_data[config["predictors"]].copy()
    y = input_data[config["label"]].copy()

    # 1st split train and test
    x_train, x_test, \
    y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 42,
        stratify = y
    )

    # 2nd split test and valid
    x_valid, x_test, \
    y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = config["valid_size"],
        random_state = 42,
        stratify = y_test
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)

    # 3. Convert to datetime
    #raw_dataset = convert_datetime(raw_dataset, config)

    # 4. Data defense for non API data
    check_data(raw_dataset, config)

    # 5. Splitting train, valid, and test set
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = split_data(raw_dataset, config)

    # 6. Save train, valid and test set
    utils.pickle_dump(x_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])

    utils.pickle_dump(x_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])

    utils.pickle_dump(x_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])

    utils.pickle_dump(raw_dataset, config["dataset_cleaned_path"])