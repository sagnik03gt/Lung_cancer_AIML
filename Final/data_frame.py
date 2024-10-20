import pandas as pd

data = pd.read_csv("StudentPerformanceFactors.csv")

# print(data)

# print(data.head())
# print(data.tail())

# print(data.isna().sum())
# print(data["Teacher_Quality"])
# new_data = data.fillna(data['Teacher_Quality'].mode())
# # print(new_data.isna().sum())
# print(new_data["Teacher_Quality"])


print(data)

for i in data.index:
    if data.loc[i,"Attendance"] > 50:
        data.loc[i, "Att_remarks"] = "Good"
print(data)











