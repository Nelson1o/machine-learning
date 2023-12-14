import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('E:\\Магистратура\\3\\Машинное обучение\\healthcare-dataset-stroke-data.csv', sep=',')
print("Данные: ")
print(df.head(10))

print("\nСтатистика по числовым показателям: ")
print(df.describe())

sample_df = df.sample(100)
print("\nПростой случайный сэмплинг: ")
print(sample_df)
# ===========================================================================================
print("\nБаланс выборки (ресэмплинг): ")
target_count = df.stroke.value_counts()
print(target_count)
print('Class 0: ', target_count[0])
print('Class 1: ', target_count[1])
print('Пропорция: ', round(target_count[0] / target_count[1], 2), ': 1')
sns.countplot(data=df, x='stroke')
plt.box(False)
plt.xlabel('Stroke 1/0', fontsize=11)
plt.ylabel('Количество наблюдений', fontsize=11)
plt.show()

shuffled_df = df.sample(frac=1, random_state=4)
CHD_df = shuffled_df.loc[shuffled_df['stroke'] == 1]
none_CHD_df = shuffled_df.loc[shuffled_df['stroke'] == 0].sample(n=250, random_state=42)
normalize_df = pd.concat([CHD_df, none_CHD_df])
target_count2 = normalize_df.stroke.value_counts()
print(target_count2)
sns.countplot(data=normalize_df, x='stroke')
plt.box(False)
plt.xlabel('Quality 1/0', fontsize=11)
plt.ylabel('Количество наблюдений', fontsize=11)
plt.show()
# ===========================================================================================
X = df.iloc[:, [0, 2, 3, 4, 8]]
y = df['stroke']

from imblearn.under_sampling import TomekLinks

tl = TomekLinks(sampling_strategy='majority')
X_res, y_res = tl.fit_resample(X, y)
print(X_res.shape)
print(X.shape)
df_new = pd.concat([X_res, y_res], axis=1)
print(df_new.shape)
sns.countplot(data=df_new, x='stroke')
plt.title('Under-sampling')
plt.box(False)
plt.xlabel('Stroke 1/0', fontsize=11)
plt.ylabel('Количество наблюдений', fontsize=11)
plt.show()

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)
print(X_sm.shape)
print(X.shape)
df_new_sm = pd.concat([X_sm, y_sm], axis=1)
print(df_new_sm.shape)
sns.countplot(data=df_new_sm, x='stroke')
plt.title('Over-sampling')
plt.box(False)
plt.xlabel('Stroke 1/0', fontsize=11)
plt.ylabel('Количество наблюдений', fontsize=11)
plt.show()

from imblearn.under_sampling import RandomUnderSampler

sampling_strategy = 0.8
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_res, y_res = rus.fit_resample(X, y)
autopct = "%.2f"
ax = y_res.value_counts().plot.pie(autopct=autopct)
_ = ax.set_title("Under-sampling")
plt.show()

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(sampling_strategy=sampling_strategy)
X_s, y_s = ros.fit_resample(X, y)
ax = y_s.value_counts().plot.pie(autopct=autopct)
_ = ax.set_title("Over-sampling")
plt.show()

sampling_strategy = "not minority"
fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_res, y_res = rus.fit_resample(X, y)
y_res.value_counts().plot.pie(autopct=autopct, ax=axs[0])
axs[0].set_title("Under-sampling")

sampling_strategy = "not majority"
ros = RandomOverSampler(sampling_strategy=sampling_strategy)
X_s, y_s = ros.fit_resample(X, y)
y_s.value_counts().plot.pie(autopct=autopct, ax=axs[1])
_ = axs[1].set_title("Over-sampling")
plt.show()
