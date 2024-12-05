import pandas as pd

def load_data():
    print("== Load Data: start ==")
    data = pd.read_excel("data/paired_data_newSim.xlsx")
    _temp = data["word_pair"].str.split(".")
    data["word1"] = _temp.apply(func=lambda x: x[0])
    data["word2"] = _temp.apply(func=lambda x: x[1][1:])

    print("== Load Data: end ==")
    return data