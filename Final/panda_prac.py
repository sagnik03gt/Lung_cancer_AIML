import pandas as pd

names = ['a', 'b', 'c', 'd']
names_dict = {'a1': 'john',
              'b1': 'sayan',
              'c1': 'amit'
              }
print(names_dict)

names_series = pd.Series(names_dict, index=[1, 2, 3, 4, 5])
print(names_series)

data = {
    'name': ['sayan''amal', 'rohan'],
    'address': ['kolkata', 'mumbai', 'chennai'],
    'age': [20, 21, 22]
}
pdata_df = pd.DataFrame(data)
print(pdata_df)
