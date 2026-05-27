# credit-scoring-eda-modeling

## 1. Ознакомление с данными

### 1.1 Первичный анализ структуры и типов данных

В файлах Training.xlsx и Test.xlsx видно, что датасет содержит как числовые зашифрованные признаки (A, B, C, D, E, F, Y, Z), так и категориальные данные (пол, регион, занятость, образование, семейное положение, количество детей, тип собственности).

```python
import pandas as pd


train_df = pd.read_excel('Training.xlsx')


print("Общая информация")
print(train_df.info())

print("\nПропущенные значения")
print(train_df.isnull().sum().sort_values(ascending=False).head(10))
```

<img width="464" height="670" alt="image" src="https://github.com/user-attachments/assets/3398052d-02ed-454a-9988-c958db59556b" />

Всего в таблице 91528 записей. Можно заметить, что не все колонки имеют 100% заполненности, например, в признаке Z пропущено более 50%, а в H - 15%. 

<img width="175" height="220" alt="image" src="https://github.com/user-attachments/assets/d554dc30-87ff-4808-8d2f-58363d2e3ae0" />


Наличие пропусков нормальная ситуация, не всегда данные заполнены и это важно иметь ввиду, так как от этого будет зависеть метод моделирования. Как именно разберу ниже.

Также данные в некоторых колонках не приведены к одному виду: колонка I (WOMAN, Man, Woman, MAN). Для алгоритма это четыре разных категории, а не две. Модель раздробит выборку на мелкие группы, что приведет к переобучению или некорректным весам. Качество предсказаний упадет, так как логическая связь будет размыта. Поэтому я приведу всё к нижнему регистру.

### 1.2 Анализ распределения целевого признака

Важно провести анализ дисбаланса классов, так как дефолты (1), как правило, происходят значительно реже, чем успешные выплаты (0).

```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

train_df = pd.read_excel('Training.xlsx')

target_counts = train_df['MARKER'].value_counts()

print("Количество объектов по классам:")
print(target_counts)

print("\nПроцентное соотношение:")
print(target_counts / len(train_df))

plt.figure(figsize=(7, 5))

ax = sns.countplot(
    x='MARKER',
    data=train_df,
    hue='MARKER',
    palette='viridis',
    legend=False
)

ax.set_yscale('log')

for p in ax.patches:
    height = p.get_height()
    ax.annotate(
        f'{int(height)}',
        (p.get_x() + p.get_width() / 2, height),
        ha='center',
        va='bottom'
    )

plt.title('Распределение целевой переменной')
plt.xlabel('Класс')
plt.ylabel('Количество клиентов')

plt.tight_layout()

plt.show()
```
<img width="694" height="496" alt="image" src="https://github.com/user-attachments/assets/f7df886d-adf9-4144-85f4-7c28e5a58ef3" />

<img width="260" height="198" alt="image" src="https://github.com/user-attachments/assets/4f9940c5-07dc-4882-a949-fcc509f3ae06" />

Анализ целевой переменной MARKER показал очень сильный дисбаланс, так как клиентов без дефолта (0) — 91 181 наблюдение (99.62%),
клиентов с дефолтом (1) — всего 347 наблюдений (0.38%).

**Влияние дисбаланса на модель**
- модель будет  предсказывать только класс большинства (0)
- модель может практически игнорировать редкий класс (1)
- ухудшается способность выявлять клиентов с высоким кредитным риском

**Как улучшить предсказание**
- использовать переменную `stratify=y`, при разделении данных на тестовую и тренировочную выборку, сохраняет одинаковую долю классов.
- применять взвешивание классов `class_weight='balanced'`, автоматически увеличивает штраф за ошибки на редком классе.
- использовать методы `oversampling` или `undersampling`, искусственно увеличивает/уменьшает количество данных.

### 1.3 Корреляционный анализ и выявление зависимостей

**Корреляционный анализ** — это метод статистического исследования, который позволяет оценить степень линейной взаимосвязи между числовыми переменными в данных.

В рамках задачи он используется для того, чтобы понять, какие признаки связаны между собой и могут дублировать друг друга, а также какие признаки потенциально влияют на целевую переменную.

Для каждой пары числовых признаков рассчитывается коэффициент корреляции Пирсона, который принимает значения от -1 до 1:
- +1 — сильная положительная связь (оба признака растут вместе)
- 0 — отсутствует линейная связь
- -1 — сильная отрицательная связь (один растёт, другой уменьшается)

Полученные значения объединяются в корреляционную матрицу, которая показывает взаимосвязи между всеми числовыми переменными одновременно.

**Корреляционный анализ применяется для:**
- выявления сильно связанных признаков
- снижения мультиколлинеарности, которая может ухудшать качество моделей
- упрощения модели за счёт удаления избыточных переменных
- первичной оценки того, какие признаки могут быть связаны с целевой переменной

```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

train_df = pd.read_excel('Training.xlsx')


features = train_df.select_dtypes(include=[np.number]).drop(columns=['ID', 'MARKER'])
corr_matrix = train_df[features.columns.tolist() + ['MARKER']].corr()

plt.figure(figsize=(12, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    square=True
)

plt.show()
```


<img width="874" height="757" alt="image" src="https://github.com/user-attachments/assets/10f104b3-6591-4d54-a160-ceb016d6160b" />

Группы A и B имеют сильную положительную связь. Они частично описывают одно и то же.

Набор D, E, F очень похожи, поэтому это классическая мультиколлинеарность.

Большинство остальных связей слабые.

Связь с целевой переменной: все значения около 0, нет сильной линейной зависимости между признаками и целевой переменной. Линейные модели здесь будут плохо работать. 

### 1.4 Анализ категориальных признаков

**Дефолт по полу**
<img width="991" height="523" alt="image" src="https://github.com/user-attachments/assets/957639a0-42f7-4c64-8084-706fe0a43981" />

**Дефолт по региону**
<img width="991" height="501" alt="image" src="https://github.com/user-attachments/assets/a05c51f5-4046-41c5-bce6-a0cbea6e72f6" />

**Дефолт по должности**
<img width="993" height="472" alt="image" src="https://github.com/user-attachments/assets/9d9e5039-3694-4b1d-a7a2-0fb4e794a672" />

Разбиение категорий в признаке “должность” содержит неоднородные и частично дублирующие группы. В частности, позиции Head/Deputy head (organiz.) и Head/Deputy head (division) логически относятся к одной смысловой категории управленческого уровня, однако разделены на отдельные классы. В целом структура признака выглядит избыточно детализированной и не обеспечивает чистого семантического разделения должностей.

**Дефолт по образованию**
<img width="988" height="433" alt="image" src="https://github.com/user-attachments/assets/b0d95271-9bed-41e3-b072-2b1ea6cf3a80" />

**Дефолт по семейному положению**
<img width="992" height="494" alt="image" src="https://github.com/user-attachments/assets/d7cc8b3d-bc7d-4368-aa83-ad31734888d3" />

**Дефолт по наличию жилья**
<img width="986" height="449" alt="image" src="https://github.com/user-attachments/assets/bd480d53-a085-490b-a8d4-eb568df3b1e9" />

**Дефолт по занятости**
<img width="994" height="484" alt="image" src="https://github.com/user-attachments/assets/d37ee508-a9d0-4e0c-8386-0516cf3679c6" />

Присутствует категория “не в паре”, которая относится не к трудовому положению, а к семейному статусу. Это нарушает семантическую однородность признака и может искажать интерпретацию зависимости дефолта от занятости.




## 2. Выполнение моделирования
Обоснование методологии:
Препроцессинг: Числовые признаки масштабируются (StandardScaler) и заполняются медианой во избежание влияния выбросов. Категориальные признаки кодируются через OneHotEncoder с заполнением пропусков модой.
Валидация: Используется StratifiedKFold (5 фолдов), чтобы сохранить пропорцию классов в обучающей и валидационной выборках.
Метрика оптимизации: Основной метрикой выбран ROC-AUC, так как он оценивает способность модели разделять классы независимо от выбранного порога вероятности, что является стандартом в банковском скоринге.
Предобработка данных (Pipeline)
```python
# Разделение признаков и таргета
X_train = train_df.drop(columns=['ID', 'MARKER'])
y_train = train_df['MARKER']
X_test = test_df.drop(columns=['ID', 'MARKER'])
y_test = test_df['MARKER']

# Списки колонок
num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

# Определение трансформеров
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])
```

Алгоритм 1: Линейная модель (Логистическая регрессия)
Метод: Логистическая регрессия с L2-регуляризацией (Ridge) для минимизации риска переобучения на мультиколлинеарных признаках. Дополнительно применяется балансировка весов классов (class_weight='balanced').
```python
from sklearn.linear_model import LogisticRegression

# Создание пайплайна
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])

# Сетка гиперпараметров
lr_param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10]
}

# Поиск по сетке
lr_cv = GridSearchCV(lr_pipeline, lr_param_grid, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
lr_cv.fit(X_train, y_train)

print(f"Лучшие параметры Логистической регрессии: {lr_cv.best_params_}")
print(f"Лучший Train ROC-AUC: {lr_cv.best_score_:.4f}")
```
Алгоритм 2: Случайный лес (Random Forest)
Метод: Ансамблевый алгоритм бэггинга над решающими деревьями. Устойчив к выбросам, не требует строго линейных зависимостей, эффективно работает с категориальными признаками после One-Hot кодирования.
```python
from sklearn.ensemble import RandomForestClassifier

# Создание пайплайна
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Сетка гиперпараметров
rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [2, 5]
}

# Поиск по сетке
rf_cv = GridSearchCV(rf_pipeline, rf_param_grid, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
rf_cv.fit(X_train, y_train)

print(f"Лучшие параметры Random Forest: {rf_cv.best_params_}")
print(f"Лучший Train ROC-AUC: {rf_cv.best_score_:.4f}")
```

Алгоритм 3: Градиентный бустинг (LightGBM)
Метод: Реализация градиентного бустинга от Microsoft, оптимизированная по скорости и объему потребляемой памяти. Последовательно строит деревья, минимизируя ошибку предыдущих, отлично улавливает сложные нелинейные паттерны.
```python
from lightgbm import LGBMClassifier

# Создание пайплайна
lgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LGBMClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), 
                                  random_state=42, verbose=-1))
])

# Сетка гиперпараметров
lgb_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__max_depth': [3, 5, 7]
}

# Поиск по сетке
lgb_cv = GridSearchCV(lgb_pipeline, lgb_param_grid, cv=StratifiedKFold(5), scoring='roc_auc', n_jobs=-1)
lgb_cv.fit(X_train, y_train)

print(f"Лучшие параметры LightGBM: {lgb_cv.best_params_}")
print(f"Лучший Train ROC-AUC: {lgb_cv.best_score_:.4f}")
```

Шаг 2.4: Сравнение моделей и финальная оценка на отложенной выборке (Test)
Выбираем лучшую модель по результатам кросс-валидации и проводим её финальное тестирование на данных из test_df.
```python
# Определение лучшей модели по итогам CV (пример выбора LightGBM)
best_model = lgb_cv.best_estimator_

# Предсказание на тестовой выборке
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Расчет финальной метрики
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

print("=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ ===")
print(f"Финальный Test ROC-AUC: {test_roc_auc:.4f}\n")
print("Отчет о классификации (Classification Report):")
print(classification_report(y_test, y_pred))

# Визуализация ROC-кривой
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) на Test')
plt.legend(loc="lower right")
plt.show()
```





