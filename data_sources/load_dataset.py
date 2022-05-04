from data_sources.get import get_indicators
import numpy as np
from pandas import DataFrame


INDICATORS_SELECTED = [
    "SP.POP.GROW",  # Population growth (annual %)
    "FP.CPI.TOTL.ZG",  # Inflation, consumer prices (annual %)
    "SP.DYN.LE00.IN",  # Life expectancy at birth, total (years)
    "NE.EXP.GNFS.ZS",  # Exports of goods and services (% of GDP)
    "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
    "SL.UEM.TOTL.ZS",  # Unemployment, total (% of total labor force) (modeled ILO estimate)
    "NV.AGR.TOTL.ZS",  # Agriculture, forestry, and fishing, value added (% of GDP)
    "EG.ELC.ACCS.ZS",  # Access to electricity (% of population)
    "AG.LND.FRST.ZS",  # Forest area (% of land area)
    "SH.DYN.MORT",  # Mortality rate, under-5 (per 1,000 live births)
    "NY.GDP.TOTL.RT.ZS",  # Total natural resources rents (% of GDP)
    "SP.DYN.TFRT.IN",  # Fertility rate, total (births per woman)
    "EN.URB.LCTY.UR.ZS",  # Population in the largest city (% of urban population)
    "TG.VAL.TOTL.GD.ZS",  # Merchandise trade (% of GDP)
    "MS.MIL.XPND.GD.ZS",  # Military expenditure (% of GDP)
]

INDICATORS_AGRICULTURE = [
    "AG.LND.ARBL.ZS",  # Arable land (% of land area)
    "AG.YLD.CREL.KG",  # Cereal yield (kg per hectare)
    "SL.AGR.EMPL.FE.ZS",  # Employment in agriculture, female (% of female employment) (modeled ILO estimate)
    "AG.CON.FERT.ZS",  # Fertilizer consumption (kilograms per hectare of arable land)
    "AG.LND.FRST.ZS",  # Forest area (% of land area)
    "AG.PRD.LVSK.XD",  # Livestock production index (2014-2016 = 100)
    "AG.LND.AGRI.ZS",  # Agricultural land (% of land area)
    "NV.AGR.TOTL.ZS",  # Agriculture, forestry, and fishing, value added (% of GDP)
    "AG.LND.ARBL.HA.PC",  # Arable land (hectares per person)
    "AG.PRD.CROP.XD",  # Crop production index (2014-2016 = 100)
    "SL.AGR.EMPL.MA.ZS",  # Employment in agriculture, male (% of male employment) (modeled ILO estimate)
    "AG.PRD.FOOD.XD",  # Food production index (2014-2016 = 100)
    "AG.LND.CROP.ZS",  # Permanent cropland (% of land area)
    "SP.RUR.TOTL.ZS",  # Rural population (% of total population)
]

INDICATORS_HEALTH = [
    "SP.DYN.LE00.IN",  # Life expectancy at birth, total (years)
    "SH.DYN.MORT",  # Mortality rate, under-5 (per 1,000 live births)
    "SP.DYN.TFRT.IN",  # Fertility rate, total (births per woman)
    "SN.ITK.DEFC.ZS",  # Prevalence of undernourishment (% of population)
    "SH.IMM.IDPT",  # Immunization, DPT (% of children ages 12-23 months)
    "SP.POP.GROW",  # Population growth (annual %)
    "SP.POP.DPND",  # Age dependency ration ($ of working-age population)
    "SH.TBS.INCD",  # Incidence of tuberculosis (per 100,000 people)
    "SH.IMM.MEAS",  # Immunization, measles (% of children ages 12-23 months)
    "SP.ADO.TFRT",  # Adolescent fertility rate (births per 1,000 women ages 15-19)
    "SP.DYN.CDRT.IN",  # Death rate, crude (per 1,000 people)
    "SP.DYN.CBRT.IN",  # Birth rate, crude (per 1,000 people)
]

INDICATORS_ECONOMY = [
    "NY.GNP.PCAP.CD",  # GNI per capita, Atlas method (current US$)
    "NE.GDI.TOTL.ZS",  # Gross capital formation (% of GDP)
    "NE.IMP.GNFS.ZS",  # Imports of goods and services (% of GDP)
    "NY.GNS.ICTR.ZS",  # Gross savings (% of GDP)
    "NV.IND.TOTL.ZS",  # Industry (including construction), value added (% of GDP) - kilka braków
    "NY.GDP.DEFL.KD.ZG",  # Inflation, GDP deflator (annual %)
    "FP.CPI.TOTL.ZG",  # Inflation, consumer prices (annual %)
    "NV.MNF.TECH.ZS.UN",  # Medium and high-tech manufacturing value added (% manufacturing value added)
    "NV.AGR.TOTL.ZS",  # Agriculture, forestry, and fishing, value added (% of GDP)
    "NE.EXP.GNFS.ZS",  # Exports of goods and services (% of GDP)
    "NY.GDP.PCAP.CD",  # GDP per capita (current US$)
    "NY.GDP.MKTP.KD.ZG",  # GDP growth (annual %)
]

ALL_INDICATORS = list(
    set(
        INDICATORS_SELECTED
        + INDICATORS_AGRICULTURE
        + INDICATORS_HEALTH
        + INDICATORS_ECONOMY
    )
)

INDICATORS_YEARS_RANGE = slice(2000, 2018)


def load_dataset(
    indicators=ALL_INDICATORS, years=INDICATORS_YEARS_RANGE, nans_threshold=2
):
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

    return df_cleared


def load_time_series(
    indicators=ALL_INDICATORS, years=INDICATORS_YEARS_RANGE, nans_threshold=2
):
    df = load_dataset(indicators, years, nans_threshold)
    time_series_dict = {}
    years = df.unstack().index.values
    countries = df.unstack().columns.get_level_values(1).unique().values

    for col in df.columns:
        time_series_dict[col] = df[col].unstack().values.T

    return time_series_dict, countries, years


def split_by_columns(dataset: DataFrame):
    frames = {}
    for column in dataset.columns:
        frames[column] = (
            dataset[column]
            .to_frame()
            .reset_index()
            .pivot(index="Year", columns="Country Name", values=column)
        )
    return frames
