from data_sources.get import get_indicators
import numpy as np


def load_dataset(indicators=None, years=slice(2000, 2020), nans_threshold=2):
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
    aggregates = np.array(
        [
            "Africa Eastern and Southern",
            "Africa Western and Central",
            "Arab World",
            "Caribbean small states",
            "Central Europe and the Baltics",
            "Early-demographic dividend",
            "East Asia & Pacific",
            "East Asia & Pacific (excluding high income)",
            "East Asia & Pacific (IDA & IBRD countries)",
            "Euro area",
            "Europe & Central Asia",
            "Europe & Central Asia (excluding high income)",
            "Europe & Central Asia (IDA & IBRD countries)",
            "European Union",
            "Fragile and conflict affected situations",
            "Heavily indebted poor countries (HIPC)",
            "High income",
            "IBRD only",
            "IDA & IBRD total",
            "IDA blend",
            "IDA only",
            "IDA total",
            "Late-demographic dividend",
            "Latin America & Caribbean",
            "Latin America & Caribbean (excluding high income)",
            "Latin America & the Caribbean (IDA & IBRD countries)",
            "Least developed countries: UN classification",
            "Low & middle income",
            "Low income",
            "Lower middle income",
            "Middle East & North Africa",
            "Middle East & North Africa (excluding high income)",
            "Middle East & North Africa (IDA & IBRD countries)",
            "Middle income",
            "North America",
            "OECD members",
            "Other small states",
            "Pacific island small states",
            "Post-demographic dividend",
            "Pre-demographic dividend",
            "Small states",
            "South Asia",
            "South Asia (IDA & IBRD)",
            "Sub-Saharan Africa",
            "Sub-Saharan Africa (excluding high income)",
            "Sub-Saharan Africa (IDA & IBRD countries)",
            "Upper middle income",
            "World",
        ]
    )
    df_nans = dataset.isnull().sum().reset_index()
    countries_with_nans = df_nans[df_nans[0] > nans_threshold]["Country Name"].unique()
    df_cleared = dataset.stack()
    df_cleared.drop(index=aggregates, level=1, inplace=True)
    df_cleared.drop(index=countries_with_nans, level=1, inplace=True)
    df_cleared = df_cleared.unstack()
    df_cleared = df_cleared.bfill().ffill()
    df_cleared = df_cleared.stack()
    df_cleared.dropna(axis=0, inplace=True)

    return df_cleared.stack()


def load_time_series():
    df = load_dataset()
    time_series_dict = {}
    years = df.unstack().index.values
    countries = df.unstack().columns.get_level_values(1).unique().values

    for col in df.columns:
        time_series_dict[col] = df[col].unstack().values.T

    return time_series_dict, countries, years
