import pandas as pd

def load_rate(path="DATA/NFEM_Rates_Data_in_Excel.csv"):
    df = pd.read_csv(path)
    df["ratedate"] = pd.to_datetime(df["ratedate"], format="%B-%d-%Y")
    df = df.sort_values("ratedate", kind="stable").set_index("ratedate")
    df = df[~df.index.duplicated(keep="first")]
    return df["closingrate"].asfreq("B").ffill()