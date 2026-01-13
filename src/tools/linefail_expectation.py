import pandas as pd

file_path = r'E:\fyp\github1\ResilienceAssessment\ResilienceAssessment\data\linefailprob.xlsx'

excel_file = pd.ExcelFile(file_path)

sheet_names = excel_file.sheet_names

row_values = [[] for _ in range(32)]

for sheet_name in sheet_names:
    df = excel_file.parse(sheet_name)
    for i in range(32):
        row = df.iloc[i].tolist()
        row_values[i].extend(row)

expectations = []
for row in row_values:
    expectation = sum(row) / len(row)
    expectations.append(expectation)

print(expectations)
