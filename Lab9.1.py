import pandas as pd

data = pd.read_csv('E:\\Магистратура\\3\\Машинное обучение\\WineQuality\\winequality-red.csv', sep=';')
print("Данные: ")
print(data.head(10))

print("\nКоличество строк и столбцов: ")
print(data.shape)

print("\nПервый столбец: ")
print(data.iloc[:, 0].head(20))
print("\nТретий столбец: ")
print(data.iloc[:, 2].head(20))
print("\nПервый и третий столбцы: ")
print(data.iloc[:, [0, 2]].head(20))

print("\n23 случайные строки: ")
print(data.sample(23))

print("\nСтатистика по числовым показателям: ")
print(data.describe())

print(data[data['alcohol'] > 10].head(20))
print("Группировка данных по столбцу 'quality': ")
print(data.groupby(by=['quality'], as_index=False).mean())

print("Множественная агрегация для нескольких столбцов: ")
df_agg = data.groupby(by=['quality'], as_index=False)[['pH', 'sulphates', 'alcohol']].agg(['mean', 'min', 'max', 'std'])
print(df_agg)