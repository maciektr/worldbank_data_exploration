from data_sources.get import get_indicators


def load_dataset(indicators=None, years=slice(2000, 2019), nans_threshold=2):
    if indicators is None:
        indicators = [
            "SP.POP.GROW",
            "FP.CPI.TOTL.ZG",
            "SP.DYN.LE00.IN",
            "NE.EXP.GNFS.ZS",
            "NY.GDP.MKTP.KD.ZG",
            "SL.UEM.TOTL.ZS",
            "NV.AGR.TOTL.ZS",
            "EG.ELC.ACCS.ZS",
            "AG.LND.FRST.ZS",
            "SH.DYN.MORT",
            "NY.GDP.TOTL.RT.ZS",
            "SP.DYN.TFRT.IN",
            "EN.URB.LCTY.UR.ZS",
            "TG.VAL.TOTL.GD.ZS",
            "MS.MIL.XPND.GD.ZS",
        ]

    df = get_indicators(indicators)
    df = df.pivot_table(
        values="Value", index="Year", columns=["Indicator Name", "Country Name"]
    )
    df_cleared = clear_dataset(df.loc[years], nans_threshold)

    return df_cleared


def clear_dataset(dataset, nans_threshold):
    df_nans = dataset.isnull().sum().reset_index()
    countries_with_nans = df_nans[df_nans[0] > nans_threshold]["Country Name"].unique()
    df_cleared = dataset.stack()
    df_cleared.drop(index=countries_with_nans, level=1, inplace=True)
    df_cleared = df_cleared.bfill().ffill()

    return df_cleared


def load_time_series():
    df = load_dataset()
    time_series_dict = {}
    years = df.unstack().index.values
    countries = df.unstack().columns.get_level_values(1).unique().values

    for col in df.columns:
        time_series_dict[col] = df[col].unstack().values.T

    return time_series_dict, countries, years
