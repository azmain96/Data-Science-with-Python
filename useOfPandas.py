import pandas as pd

pd.Series
print('=== Series() shows indices and datatype of a object === ')
animal = ['Tiger', 'Bear', 'Moose']
print(pd.Series(animal))
number = [1, 2, 3, 4, 5, 5, 8, 8]
print(pd.Series(number))
print(set(number))
animals = ['Tiger', 'Hours', 'Cat', None]
print(pd.Series(animals))
numbers = [1, 2, 3, 4, 5, None]
print(pd.Series(numbers))

import numpy as np

c = np.nan is None
print(c)

sports = {
    'Cricket': 'Bangladesh',
    'Golf': 'Scotland',
    'Sumo': 'Japan',
    'Football': 'Germany'
}

s = pd.Series(sports)
print(s)

anm = pd.Series(['Tiger', 'Bear', 'Dog'], index=['Bangladesh', 'America', 'German'])
print(anm)

sports = {
    'Cricket': 'Bangladesh',
    'Golf': 'Scotland',
    'Sumo': 'Japan',
    'Football': 'Germany'
}

s = pd.Series(sports, index=['Cricket', 'Sumo'])
print(s)
print('=======================')
print(pd.Series(sports, index=['Cricket', 'Golf', 'Football', 'Hockey']))
print('=====loc and iloc used to show values according to index=====')
s = pd.Series(sports)
print(s, '\n')
# iloc and loc they are not method they are attributes. Those are indexing operator
print(s.iloc[2])
print(s.loc['Golf'])
print(s[3])
print(s[0])
print(s.iloc[0])
print(s['Sumo'])
# print(s['Japan']) error
print('=========================')
sp = {
    99: 'Bhutan',
    100: 'Scotland',
    101: 'Thailand',
    102: 'Korea',
    103: 'Netherlands'
}
d = pd.Series(sp)
print(d)
print(d.iloc[0])
print('==========================')
srs = pd.Series([10, 20.5, 80, 102, 120.8, 3.00])
print(srs)
# print(srs-1)
total = 0
for i in srs:
    total += i
print(total)

total = np.sum(srs - 1)
print(total)

print('=======================')

randSeries = pd.Series(np.random.randint(10, 1000, 500))
print(randSeries.head())
# print(randSeries)
print('Length of random series:', len(randSeries))
print(np.sum(randSeries))
print(np.sum(randSeries.head()))

a = pd.Series([1, 2, 3])
a.loc['Animal'] = 'Bears'
a['Bird'] = 'Kingfisher'
print(a)
print(len(a))
print('======================')

original_sports = pd.Series({
    'Archery': 'Bhutan',
    'Golf': 'Scotland',
    'Sumo': 'Japan',
    'Football': 'Germany'
})
cricket_loving_country = pd.Series(
    ['Australia', 'India', 'Pakistan', 'England'],
    index=['Cricket', 'Cricket', 'Cricket', 'Cricket']
)
all_countries = original_sports.append(cricket_loving_country)

print(all_countries)
print(cricket_loving_country)
print('all countries playing cricket:\n', all_countries.loc['Cricket'])
print('== DataFrame is a 2 dimensional data structure. It is like a spreadsheet or SQL table == ')
purchase_1 = pd.Series({
    'Name': 'Chris',
    'Item Purchase': 'Dog food',
    'Cost': 22.50})
purchase_2 = pd.Series({
    'Name': "Kevyn",
    'Item Purchase': 'Kitty Litter',
    'Cost': 2.50})
purchase_3 = pd.Series({
    'Name': 'Vinod',
    'Item Purchase': "Bird Seed",
    'Cost': 5.0})
purchase_4 = pd.Series({
    'Name': 'Tazrian',
    'Item Purchase': 'Plant Seeds',
    'Cost': 3.99
})

df = pd.DataFrame([purchase_1, purchase_2, purchase_3, purchase_4], index=['Store1', 'Store1', 'Store2', 'Store2'])
print(df)
print('\n')
print(df.head())
print('\nAll data from store 1:\n', df.loc['Store1'])
print(type(df.loc['Store2']))
print('\nstore 1 cost')
print(df.loc['Store1', "Cost"])
print('\n')
print(df.loc['Store2', "Item Purchase"])
print(df.loc[:, ['Name', 'Item Purchase']])  # ':' means information of all stores
print('\n')
print(df.loc['Store1']['Cost'])
print('==============================')
TransData = df.T
print(TransData)  # Transpose Data
print('\n')
print(TransData.loc['Name'])
print(TransData.loc['Cost']['Store1'])
print('=== Dropping store2 ===')
d = df.drop('Store2')
print(d)
print('====== copy data ======')
copy_df = df.copy()
print(copy_df)
print('====== deleting column =====')
del copy_df['Name']
print(copy_df)
print('=======================')
del copy_df['Cost']
print(copy_df)

df['Location'] = None
print(df)
print('=====================')

df['Email'] = ['x@gmai.com', 'y@gmai.com', 'a@gmai.com', 'b@yahoo.com']
print(df)
print('====================')

costs = df['Cost']
print(costs, '\n')
costs += 1
print(costs)

olympicData = pd.read_csv('olympics.csv')
print(olympicData.head())
print(olympicData.columns)

olympicData = pd.read_csv('olympics.csv', index_col=0, skiprows=1)
print(olympicData.head())

print('======================')
print(olympicData.columns)
for col in olympicData.columns:
    # print(col,0,col[4:])
    if col[:2] == '01':
        olympicData.rename(columns={col: 'Gold' + col[4:]}, inplace=True)
    if col[:2] == '02':
        olympicData.rename(columns={col: 'Silver' + col[4:]}, inplace=True)
    if col[:2] == '03':
        olympicData.rename(columns={col: 'Bronze' + col[4:]}, inplace=True)
    if col[:1] == '№':
        olympicData.rename(columns={col: '#' + col[4:]}, inplace=True)

print(olympicData.head())

print(df)
print(df.columns)

for i in df.columns:
    if i[:2] == 'Co':
        df.rename(columns={i: '$'+i[:]}, inplace=True)  # 'columns' is a builtin parameter of pandas. inplace might be True


print(df)
print('\n')
print(olympicData['Gold'])
print('================================')
print(olympicData['Gold']>0)
print('================#####================')
only_gold = olympicData.where(olympicData['Gold'] > 0)
print(only_gold[:15])
print('====where() function takes conditions and return a new dataframe according to condition=====')
print(only_gold['Gold'].count())
print(olympicData['Gold'].count())
bronze = olympicData['Bronze']<0
print(bronze)
only_bronze = olympicData.where((olympicData['Bronze'] > 0) &
                                (olympicData["Gold"] == 0) & (olympicData['Silver'] == 0))
print(only_bronze.head())
print(only_bronze['Bronze'].count())
print(olympicData['Bronze'].count())
print('============== drop() function dropped NaN value from the dataset =============')
only_bronze = only_bronze.dropna()
print(only_bronze)

lnth = len(olympicData[(olympicData['Gold'] > 0) & (olympicData['Gold.1'] > 0)])
le = len(olympicData[(olympicData['Gold'] > 0)])
print(lnth)
print(le)
x = olympicData[(olympicData['Gold'] > 0) & (olympicData['Gold.1'] ==0)]
print(x)

print('===================== INDEXING DATAFRAMES =======================\n')

olympicData['country'] = olympicData.index  # Take a new column and moved existing indices out there.
olympicData = olympicData.set_index('Gold')  # Made Gold column as indices
print(olympicData.head())

olympicData = olympicData.reset_index()  # Added numerical indices like 0 up to dataset rows.
print('\n', olympicData.head())
print(len(olympicData))
print(' ============================= ==================== ')
cencus = pd.read_csv('census.csv')
print(cencus.head())
print(' ================= Finding unique value ==================== ')
print(cencus['SUMLEV'].unique())
print(cencus[cencus['SUMLEV'] == 40])
print(cencus['REGION'].unique())
print(cencus[cencus['REGION'] == 4])
print(cencus[(cencus['SUMLEV'] == 50) & (cencus['REGION'] == 3)])

columns_to_keep = [
                   'STNAME',
                   'CTYNAME',
                   'BIRTHS2010',
                   'BIRTHS2011',
                   'BIRTHS2012',
                   'BIRTHS2013',
                   'BIRTHS2014',
                   'BIRTHS2015',
                   'POPESTIMATE2010',
                   'POPESTIMATE2011',
                   'POPESTIMATE2012',
                   'POPESTIMATE2013',
                   'POPESTIMATE2014',
                   'POPESTIMATE2015'
]

cencus = cencus[columns_to_keep]
print(cencus.head())

cencus = cencus.set_index(['STNAME', 'CTYNAME'])
print(cencus[:75])
print(cencus.loc['Michigan', 'Washtenaw County'])
print(cencus.loc[[('Michigan', 'Washtenaw County'), ('Michigan', 'Wayne County')]])
print('=========================log dataset===============================')
log = pd.read_csv('log.csv')
print(log.head(),'\n==================================================================')
log = log.set_index('time')
log = log.sort_index()
print(log)
print('============ set time and user as indices ==============')
log = log.reset_index()
log = log.set_index(['time', 'user'])
print(log)

log = log.fillna(method = 'ffill')
print('\n', log.head())

olympicData = olympicData.set_index('country')
names_idx = olympicData.index.str.split('\s\(')
olympicData.index = names_idx.str[0]

u = olympicData['Gold'].unique()
print(olympicData)
print(u)
cn = olympicData[olympicData['Gold'] == 976].index.values
print(cn)
print(olympicData['Gold'].argmax())
print(abs(olympicData['Gold'] - olympicData['Gold.1']))
print('===================================')

points = olympicData['Gold.2']*3 + olympicData['Silver.2']*2 + olympicData['Bronze.2']
print('\n', points)

# Which state has the most counties in it?
# (hint: consider the sumlevel key carefully! You'll need this for future questions too...)
# This function should return a single string value.
cencus = pd.read_csv('census.csv')
print(cencus.head())
new_df = cencus[cencus['SUMLEV'] == 50]
mostCountries = new_df.groupby('STNAME').count()['SUMLEV'].idxmax()
print(mostCountries)

df = pd.DataFrame({'A': [0, 1, 2, 3, 4],
                   'B': [5, 0, 7, 8, 9],
                   'C': ['a', 0, 'c', 'd', 'e']})
print(df)
print(df.replace(0, 10))

