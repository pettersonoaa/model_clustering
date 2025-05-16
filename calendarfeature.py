from datetime import date, timedelta, datetime
from dateutil.easter import easter
import pandas as pd
import numpy as np

def holiday_feature(start_date=None, 
                    end_date=None, 
                    weekdays=None, 
                    workingdays=None,
                    lags=None, 
                    leads=None):

    if start_date is None:
        start_date = (date.today() - timedelta(days=10*365))
    if end_date is None:
        end_date = (date.today() + timedelta(days=10*365))

    start = (start_date - timedelta(days=365))
    end = (end_date + timedelta(days=365))

    date_range = pd.date_range(start=start, end=end, freq='D')
    data = pd.DataFrame(date_range, columns=['date'])

    fixed_holidays = [
        {'name': 'anonovo','month': 1,'day': 1},
        {'name': 'aniversariosp','month': 1,'day': 25},
        {'name': 'aniversariorj','month': 3,'day': 1},
        {'name': 'tiradentes','month': 3,'day': 21},
        {'name': 'diadotrabalho','month': 5,'day': 1},
        {'name': 'diadosnamorados','month': 6,'day': 12},
        {'name': 'saojoao','month': 6,'day': 24},
        {'name': 'revolucaoconst','month': 7,'day': 9},
        {'name': 'independencia','month': 9,'day': 7},
        {'name': 'aparecida','month': 10,'day': 12},
        {'name': 'servidorpublico','month': 10,'day': 28},
        {'name': 'finados','month': 11,'day': 2},
        {'name': 'proclamacao','month': 11,'day': 15},
        {'name': 'consciencianegra','month': 11,'day': 20},
        {'name': 'natal','month': 12,'day': 25},
    ]
    for holiday in fixed_holidays:
        data['calendar_holiday_' + holiday['name']] = data['date'].apply(
            lambda x: 1.0 if (x.month == holiday['month'] and x.day == holiday['day']) else 0.0
        )
    
    easter_holidays = [
        {'name': 'carnaval','delta_days': -47},
        {'name': 'paixaocristo','delta_days': -2},
        {'name': 'pascoa','delta_days': 0},
        {'name': 'corpuschristi','delta_days': 60},
    ]
    for holiday in easter_holidays:
        data['calendar_holiday_' + holiday['name']] = data['date'].apply(
            lambda x: 1.0 if (x == (easter(x.year) + timedelta(days=holiday['delta_days']))) else 0.0
        )
    
    def black_friday(year):
        first_november = date(year, 11, 1)
        days_to_thursday = (3 - first_november.weekday()) % 7
        first_thursday = first_november + timedelta(days=days_to_thursday)
        thanksgiving_date = first_thursday + timedelta(days=21)
        return thanksgiving_date + timedelta(days=1)
    data['calendar_holiday_blackfriday'] = data['date'].apply(
        lambda x: 1.0 if (x == black_friday(x.year)) else 0.0
    )

    def dia_dos_pais(year):
        first_august = date(year, 8, 1)
        days_to_sunday = (6 - first_august.weekday()) % 7
        first_sunday = first_august + timedelta(days=days_to_sunday)
        return first_sunday + timedelta(weeks=2)
    data['calendar_holiday_diadospais'] = data['date'].apply(
        lambda x: 1.0 if (x == dia_dos_pais(x.year)) else 0.0
    )

    def dia_das_maes(year):
        first_may = date(year, 5, 1)
        days_to_sunday = (6 - first_may.weekday()) % 7
        first_sunday = first_may + timedelta(days=days_to_sunday)
        return first_sunday + timedelta(weeks=1)
    data['calendar_holiday_diadasmaes'] = data['date'].apply(
        lambda x: 1.0 if (x == dia_das_maes(x.year)) else 0.0
    )

    data = data.set_index('date')
    columns = {
        'holidays': data.columns.to_list()
    }

    # add weekday versions of holidays
    columns['weekdays'] = []
    if weekdays:
        for holiday in fixed_holidays:
            for weekday in range(7):
                col_name = 'calendar_holiday_' + holiday['name'] + '_wd' + str(weekday)
                data[col_name] = 0.0
                data.loc[(data['calendar_holiday_' + holiday['name']] == 1.0) & (data.index.weekday == weekday), col_name] = 1.0
        columns['weekdays'] = [col for col in data.columns.to_list() if '_wd' in col]

    # add working days
    columns['workingdays'] = []
    if workingdays:
        data['calendar_workingday'] = 1.0
        data.loc[data.index.weekday >= 5, 'calendar_workingday'] = 0.0
        for holiday in columns['holidays']:
            data.loc[data[holiday] == 1.0, 'calendar_workingday'] = 0.0
        columns['workingdays'] = ['calendar_workingday']        

    # adding lags
    columns['lags'] = []
    if lags:
        for lag in range(1, lags + 1):
            laged_data = data[columns['holidays']+columns['weekdays']+columns['workingdays']].copy()
            laged_data = laged_data.shift(lag)
            laged_data.columns = [f"{col}_pre{lag}" for col in laged_data.columns]
            data = data.merge(laged_data, on='date', how='left')
            columns['lags'] += laged_data.columns.to_list()
    
    columns['leads'] = []
    if leads:
        for lead in range(1, leads + 1):
            leaded_data = data[columns['holidays']+columns['weekdays']+columns['workingdays']].copy()
            leaded_data = leaded_data.shift(lead)
            leaded_data.columns = [f"{col}_pos{lead}" for col in leaded_data.columns]
            data = data.merge(leaded_data, on='date', how='left')
            columns['leads'] += leaded_data.columns.to_list()
    
    data = data[(data.index.date>=start_date) & (data.index.date<=end_date)]
    return data, columns


def seasonal_features(start_date=None, 
                      end_date=None, 
                      dayofweek=True, 
                      dayofweekpermonth=True, 
                      dayofmonth=True, 
                      monthofyear=True, 
                      breakpoint_date=None):
    
    if start_date is None:
        start_date = (date.today() - timedelta(days=10*365))
    if end_date is None:
        end_date = (date.today() + timedelta(days=10*365))

    start = (start_date - timedelta(days=365))
    end = (end_date + timedelta(days=365))

    date_range = pd.date_range(start=start, end=end, freq='D')
    data = pd.DataFrame(date_range, columns=['date'])

    breakpoint_date = datetime.strptime(breakpoint_date, "%Y-%m-%d").date() if breakpoint_date else None

    if dayofweek:
        if breakpoint_date:
            data['calendar_seas_dayofweek'] = None
            mask = data['date'].dt.date >= breakpoint_date
            data.loc[mask, 'calendar_seas_dayofweek'] = data.loc[mask, 'date'].dt.dayofweek
        else:
            data['calendar_seas_dayofweek'] = data['date'].dt.dayofweek
        
    if dayofweekpermonth:
        if breakpoint_date:
            data['calendar_seas_dayofweekpermonth'] = None
            mask = data['date'].dt.date >= breakpoint_date
            data.loc[mask, 'calendar_seas_dayofweekpermonth'] = data.loc[mask, 'date'].dt.dayofweek.astype(str) + '_' + data['date'].dt.month.astype(str)
        else:
            data['calendar_seas_dayofweekpermonth'] = data.loc[mask, 'date'].dt.dayofweek.astype(str) + '_' + data['date'].dt.month.astype(str)

    if dayofmonth:
        if breakpoint_date:
            data['calendar_seas_dayofmonth'] = None
            mask = data['date'].dt.date >= breakpoint_date
            data.loc[mask, 'calendar_seas_dayofmonth'] = data.loc[mask, 'date'].dt.day
        else:
            data['calendar_seas_dayofmonth'] = data['date'].dt.day

    if monthofyear:
        if breakpoint_date:
            data['calendar_seas_monthofyear'] = None
            mask = data['date'].dt.date >= breakpoint_date
            data.loc[mask, 'calendar_seas_monthofyear'] = data.loc[mask, 'date'].dt.month
        else:
            data['calendar_seas_monthofyear'] = data['date'].dt.month
    
    data = data.set_index('date')   

    data = pd.get_dummies(data).astype(np.float32)

    data = data[(data.index.date>=start_date) & (data.index.date<=end_date)]

    columns = {
        'dayofweek': [col for col in data.columns if '_dayofweek_' in col],
        'dayofweekpermonth': [col for col in data.columns if '_dayofweekpermonth_' in col],
        'dayofmonth': [col for col in data.columns if '_dayofmonth_' in col],
        'monthofyear': [col for col in data.columns if '_monthofyear_' in col]
    }

    return data, columns

print(holiday_feature(weekdays=False, workingdays=True,lags=2,leads=2))
print(seasonal_features(breakpoint_date='2023-01-01'))


