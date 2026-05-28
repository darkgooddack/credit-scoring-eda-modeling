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

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_excel('Training.xlsx')
test_df = pd.read_excel('Test.xlsx')

categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()

train_df['I'] = train_df['I'].astype(str).map(lambda x: x.lower().strip())
test_df['I'] = test_df['I'].astype(str).map(lambda x: x.lower().strip())


analysis_features = ['I', 'K', 'M', 'N', 'O', 'Q', 'S', ]

for col in analysis_features:
    df_grouped = (
        train_df.groupby(col)['MARKER']
        .agg(mean='mean', count='count')
        .reset_index()
        .sort_values('mean', ascending=False)
    )

    plt.figure(figsize=(10, 4))

    ax = sns.barplot(
        data=df_grouped,
        x=col,
        y='mean'
    )

    plt.xticks(rotation=45, ha='right')

    for i, row in enumerate(df_grouped.itertuples()):
        ax.text(
            i,
            row.mean,
            f"{row.mean:.2f}\n(n={row.count})",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.show()
```


## 2. Выполнение моделирования

Перед подачей данных в модели необходимо устранить аномалии, выявленные на этапе EDA, чтобы избежать переобучения и искажения весов.

**Чистка категориальных признаков**
- Привести пол к man, woman.
- Объединить Head/Deputy head (organiz.) и Head/Deputy head (division) в признаке "Должность" в Management.
- Перенести категорию "не в паре" в семейный статус, заполнение корректным значением или выделение в Unknown.

**Обработка пропусков**
- Для линейных моделей заполню медианой.

**Метрика оптимизации** 
- Основной метрикой выбран ROC-AUC, так как он оценивает способность модели разделять классы независимо от выбранного порога вероятности, что является стандартом в банковском скоринге.


## Модель 1: Линейная модель (Логистическая регрессия с регуляризацией)

- Удаление одного из признаков из пар с высокой мультиколлинеарностью (оставить только D из тройки D, E, F, убрать B из пары A, B), так как избыточные признаки ломают стабильность весов логистической регрессии.
- Параметр class_weight='balanced'.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


train_df = pd.read_excel('Training.xlsx')
test_df = pd.read_excel('Test.xlsx')


def clean_credit_data(df):
    df_clean = df.copy()

    if 'I' in df_clean.columns:
        df_clean['I'] = df_clean['I'].astype(str).str.lower().str.strip()

    pos_col = 'M'
    if pos_col in df_clean.columns:
        df_clean[pos_col] = df_clean[pos_col].astype(str).str.strip()
        management_mask = df_clean[pos_col].isin(['Head/Deputy head (organiz.)', 'Head/Deputy head (division)'])
        df_clean.loc[management_mask, pos_col] = 'Management'

    employment_col = 'S'
    marital_col = 'P'

    if employment_col in df_clean.columns:
        df_clean[employment_col] = df_clean[employment_col].astype(str).str.strip()

        anomaly_mask = df_clean[employment_col] == 'No couple'
        df_clean.loc[anomaly_mask, employment_col] = 'Unknown'

        if marital_col in df_clean.columns and anomaly_mask.any():
            df_clean.loc[anomaly_mask, marital_col] = 'Single/unmarried'

    features_to_drop = ['E', 'F', 'B']
    df_clean = df_clean.drop(columns=features_to_drop, errors='ignore')

    return df_clean


train_processed = clean_credit_data(train_df)
test_processed = clean_credit_data(test_df)


X_train_full = train_processed.drop(columns=['ID', 'MARKER'], errors='ignore')
y_train_full = train_processed['MARKER']


X_test = test_processed.drop(columns=['ID', 'MARKER'], errors='ignore')
y_test = test_processed['MARKER']


num_features = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train_full.select_dtypes(exclude=[np.number]).columns.tolist()

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

lr_model = LogisticRegression(
    class_weight='balanced',
    max_iter=2000,
    random_state=42,
    solver='lbfgs'
)

pipeline_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', lr_model)
])


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

train_scores = []
validation_scores = []


for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_full, y_train_full), 1):
    X_tr, X_va = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr, y_va = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    pipeline_lr.fit(X_tr, y_tr)

    y_tr_pred = pipeline_lr.predict_proba(X_tr)[:, 1]
    y_va_pred = pipeline_lr.predict_proba(X_va)[:, 1]

    train_scores.append(roc_auc_score(y_tr, y_tr_pred))
    validation_scores.append(roc_auc_score(y_va, y_va_pred))

print(f"\n Средний Train ROC-AUC: {np.mean(train_scores):.4f}")
print(f"\n Средний Validation ROC-AUC: {np.mean(validation_scores):.4f}")


print("\n Проверка на  выборке Test")


pipeline_lr.fit(X_train_full, y_train_full)

# Предсказание вероятностей дефолта для тестовой выборки
y_test_pred_proba = pipeline_lr.predict_proba(X_test)[:, 1]
y_test_pred_class = pipeline_lr.predict(X_test)

# Расчет финального ROC-AUC на тесте
test_auc = roc_auc_score(y_test, y_test_pred_proba)
gini_index = 2 * test_auc - 1

print(f"\nФинальный TEST ROC-AUC: {test_auc:.4f}")
print(f"\nКоэффициент Gini на тесте: {gini_index:.4f}")

print("\nМатрица ошибок на Test:")
print(confusion_matrix(y_test, y_test_pred_class))

print("\nДетальный отчет по метрикам классификации на Test:")
print(classification_report(y_test, y_test_pred_class, target_names=['Надежные (0)', 'Дефолтные (1)']))
```


<img width="471" height="432" alt="image" src="https://github.com/user-attachments/assets/cdabf60d-c73c-4b7c-b619-fd2dd6522f87" />


**Оценка разделяющей способности (ROC-AUC и Gini)**

**ROC-AUC** — это метрика качества ранжирования бинарного классификатора. Насколько хорошо модель умеет отличать класс 1 от класса 0. 

Значение 0.8698 означает, что в 87% случаев модель присваивает реальному дефолтному клиенту более высокий уровень риска, чем надежному. Для очень хороший результат.

**Gini** — это производная метрика от ROC-AUC, широко используемая в банковском скоринге.

Метрика Джини на уровне 74% в реальном банковском секторе оценивается как **отличная**. Модели с Джини выше 60% считаются высокоэффективными и допускаются до эксплуатации в скоринговых системах коммерческих банков.


**Подтверждение стабильности модели (Отсутствие переобучения)**

Сравнение метрик на разных этапах показывает методологическую корректность пайплайна предобработки:

* **Train ROC-AUC:** 0.8760 (обучение)
* **Validation ROC-AUC:** 0.8508 (кросс-валидация)
* **Test ROC-AUC:** 0.8698 (абсолютно новые данные из Test.xlsx)

Метрика на тесте практически совпадает с метрикой на кросс-валидации (и даже незначительно выше, что указывает на репрезентативность тестовой выборки). Разница с обучением минимальна. Это доказывает, что модель не переобучилась, а выявила реальные рыночные закономерности, которые стабильно работают на новых клиентах.

**Анализ матрицы ошибок и бизнес-метрик**

Матрица ошибок показывает физическое распределение предсказаний на 38 405 клиентах из тестовой выборки:

* **True Negative (29 970):** Модель верно определил надежных клиентов и одобрила им кредит.
* **True Positive (119):** Модель вовремя распознала дефолтных клиентов и заблокировала выдачу.
* **False Negative (27):** Модель пропустила дефолтных клиентов, одобрив им кредит (ошибка 1-го рода).
* **False Positive (8 289):** Модель перестраховалась и отказала надежным клиентам (ошибка 2-го рода).

**Метрики класса «Дефолтные»**

* **Recall (Полнота) = 0.82 (82%):** Ключевая метрика для управления рисками. Модель смогла успешно перехватить и предотвратить **82% всех потенциальных дефолтов** (119 из 146). Банк защищен от критических невозвратов.
* **Precision (Точность) = 0.01 (1%):** Из всех клиентов, которых модель пометила как "рискованные", реальный дефолт допустил только 1%. Столь низкое значение — это прямое следствие экстремального дисбаланса в исходных данных (дефолты составляют ничтожную долю от портфеля). Чтобы поймать 82% дефолтов, алгоритму с параметром `class_weight='balanced'` приходится сильно занижать порог одобрения, из-за чего в зону риска попадает много надежных заемщиков.

**Вывод**

Логистическая регрессия с ручным устранением мультиколлинеарности и балансировкой весов задала высокую планку качества. Модель отлично минимизирует кредитные риски, но создает высокую нагрузку на верификацию или приводит к упущенной прибыли из-за ложных отказов. 






## Модель 2: Случайный лес (Random Forest)

Устойчив к мультиколлинеарности, не требует масштабирования данных, хорошо улавливает нелинейные зависимости первого порядка.

- Использование class_weight='balanced_subsample'.
- Ограничение max_depth и min_samples_leaf для предотвращения переобучения на шумах.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


train_df = pd.read_excel('Training.xlsx')
test_df = pd.read_excel('Test.xlsx')


def clean_credit_data_for_trees(df):
    df_clean = df.copy()

    if 'I' in df_clean.columns:
        df_clean['I'] = df_clean['I'].astype(str).str.lower().str.strip()

    pos_col = 'M'
    if pos_col in df_clean.columns:
        df_clean[pos_col] = df_clean[pos_col].astype(str).str.strip()
        management_mask = df_clean[pos_col].isin(['Head/Deputy head (organiz.)', 'Head/Deputy head (division)'])
        df_clean.loc[management_mask, pos_col] = 'Management'

    employment_col = 'S'
    marital_col = 'P'

    if employment_col in df_clean.columns:
        df_clean[employment_col] = df_clean[employment_col].astype(str).str.strip()
        anomaly_mask = df_clean[employment_col] == 'No couple'
        df_clean.loc[anomaly_mask, employment_col] = 'Unknown'

        if marital_col in df_clean.columns and anomaly_mask.any():
            df_clean.loc[anomaly_mask, marital_col] = 'Single/unmarried'

    return df_clean


train_processed = clean_credit_data_for_trees(train_df)
test_processed = clean_credit_data_for_trees(test_df)

X_train_full = train_processed.drop(columns=['ID', 'MARKER'], errors='ignore')
y_train_full = train_processed['MARKER']

X_test = test_processed.drop(columns=['ID', 'MARKER'], errors='ignore')
y_test = test_processed['MARKER']


num_features = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X_train_full.select_dtypes(exclude=[np.number]).columns.tolist()


num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])


rf_model = RandomForestClassifier(
    n_estimators=300,  # Оптимальное количество деревьев для стабильного ансамбля
    max_depth=8,  # Ограничение глубины для предотвращения зазубривания шумов
    min_samples_leaf=15,  # Минимальное количество объектов в листе для стабильности
    class_weight='balanced_subsample',  # Динамическая балансировка классов на уровне подвыборок
    random_state=42,
    n_jobs=-1  # Использование всех ядер процессора для ускорения вычислений
)

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_scores = []
validation_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_full, y_train_full), 1):
    X_tr, X_va = X_train_full.iloc[train_idx], X_train_full.iloc[val_idx]
    y_tr, y_va = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    pipeline_rf.fit(X_tr, y_tr)

    y_tr_pred = pipeline_rf.predict_proba(X_tr)[:, 1]
    y_va_pred = pipeline_rf.predict_proba(X_va)[:, 1]

    train_auc = roc_auc_score(y_tr, y_tr_pred)
    val_auc = roc_auc_score(y_va, y_va_pred)

    train_scores.append(train_auc)
    validation_scores.append(val_auc)
    print(f"Фолд {fold}: Train ROC-AUC = {train_auc:.4f} | Validation ROC-AUC = {val_auc:.4f}")

print(f"\n Средний Train ROC-AUC: {np.mean(train_scores):.4f}")
print(f" Средний Validation ROC-AUC: {np.mean(validation_scores):.4f}")


pipeline_rf.fit(X_train_full, y_train_full)


y_test_pred_proba = pipeline_rf.predict_proba(X_test)[:, 1]
y_test_pred_class = pipeline_rf.predict(X_test)


test_auc = roc_auc_score(y_test, y_test_pred_proba)
gini_index = 2 * test_auc - 1

print(f"\nФинальный TEST ROC-AUC: {test_auc:.4f}")
print(f"\nКоэффициент Gini на тесте: {gini_index:.4f}")

print("\nМатрица ошибок на Test:")
print(confusion_matrix(y_test, y_test_pred_class))

print("\nДетальный отчет по метрикам классификации на Test:")
print(classification_report(y_test, y_test_pred_class, target_names=['Надежные (0)', 'Дефолтные (1)']))
```
<img width="498" height="477" alt="image" src="https://github.com/user-attachments/assets/d72d0ca6-7eac-406a-b301-5bc4a52d80d9" />


### Сравнительный анализ результатов Модели 1 и Модели 2

#### 1. Оценка обобщающей способности (ROC-AUC и Gini)

* **Качество ранжирования на тесте:** Случайный лес показал более высокий **Test ROC-AUC: 0.8724** (против 0.8698 у регрессии). Соответственно, индекс Джини вырос до **0.7447**. Это подтверждает, что нелинейный ансамбль, используя полный набор признаков, эффективнее разделяет клиентов по степени риска.
* **Выявление переобучения:** У Случайного леса зафиксирован значительный разрыв между обучением и валидацией: `Train ROC-AUC = 0.9765`, а `Validation ROC-AUC = 0.8484`. Модель почти идеально запомнила тренировочный датасет. Однако жесткая регуляризация (ограничение глубины `max_depth=8` и размера листа `min_samples_leaf=15`) уберегла ансамбль от переобучения: на отложенном тесте модель показала стабильный результат.

#### 2. Сравнение бизнес-метрик и матриц ошибок

Физическое распределение одобренных и отклоненных заявок на тестовой выборке (38 405 клиентов) отражает явное различие в стратегиях моделей.

| Метрика на тесте | Модель 1: Логистическая регрессия | Модель 2: Случайный лес | Бизнес-эффект от смены модели |
| :--- | :---: | :---: | :--- |
| **Правильно одобрено (True Negative)** | 29 970 клиентов | **35 442 клиента** | **+ 5 472 новых заемщика**, которые потенциально принесут банку процентный доход. |
| **Ложные отказы (False Positive)** | 8 289 клиентов | **2 817 клиентов** | **Снижение ложных отказов в 3 раза**. Банк сохраняет лояльность клиентской базы и снижает упущенную прибыль. |
| **Пропущенные дефолты (False Negative)**| **27 клиентов** | 63 клиента | Увеличение дефолтных выдач на 36 заемщиков (рост кредитных потерь). |
| **Перехваченные дефолты (True Positive)**| **119 клиентов** | 83 клиента | Снижение эффективности прямого удержания невозвратов на данном пороге. |
| **Recall (Полнота класса 1)** | **0.82 (82%)** | 0.57 (57%) | Логистическая регрессия защищает от дефолтов лучше на 25%. |
| **Precision (Точность класса 1)** | 0.01 (1%) | **0.03 (3%)** | Точность прогноза риска (концентрация дефолтов в зоне отказа) выросла в 3 раза. |

#### 3. Анализ

Каждый из исследованных алгоритмов предлагает банку определенную бизнес-стратегию:

1. **Консервативная стратегия (Логистическая регрессия):** Модель ориентирована на максимальное снижение кредитного риска (`Recall = 82%`). Она минимизирует прямые убытки от невозвратов, но платит за это высоким объемом ложных тревог (8 289 отказов благонадежным клиентам). Данный подход оправдан в периоды экономической нестабильности, когда приоритетом является сохранение ликвидности.
2. **Коммерческая стратегия (Случайный лес):** Модель сокращает количество ложных отказов (с 8.2 тысяч до 2.8 тысяч), позволяя одобрить кредиты более чем 5.4 тысячам надежных клиентов. Точность модели возрастает до 3%. Обратной стороной становится рост пропущенных дефолтов (63 вместо 27).

#### Вектор развития: Модель 3 (Градиентный бустинг)

Текущая дилемма формирует задачу для следующего этапа моделирования. Применение **Модели 3 (Градиентный бустинг, LightGBM/CatBoost)** должно совместить преимущества обоих подходов. За счет последовательного исправления ошибок предыдущих деревьев бустинг направлен на восстановление высокого уровня перехвата дефолтов (`Recall` на уровень регрессии $\approx 80\%$) при сохранении низкого потока ложных отказов (`False Positive` на уровне случайного леса $\approx 2800$).



## Модель 3: Градиентный бустинг (CatBoost)

Позволяет эффективно находить сложные комбинации признаков. CatBoost отлично работает с категориальными переменными.

- Параметр scale_pos_weight (отношение количества 0 к количеству 1) для компенсации дисбаланса.
- Тюнинг гиперпараметров (learning_rate, l2_leaf_reg).


### Анализ результатов

#### 1. Интерпретация метрик Модели 3 (CatBoost)

* **Качество ранжирования (Test ROC-AUC = 0.8437, Gini = 0.6874):** Модель демонстрирует стабильно высокий уровень предсказательной силы. Коэффициент Джини на уровне 68.74% превышает стандартный банковский порог эффективности 60%, что подтверждает применимость модели.
* **Поведение на матрице ошибок:** Из 146 реальных дефолтных заемщиков CatBoost успешно идентифицировал 69 (`Recall = 0.47`), допустив при этом всего 2010 ложных отказов хорошим клиентам. 


### Финальная сводная таблица моделей на тестовой выборке

Для наглядности все ключевые метрики трех построенных моделей объединены в общую структуру:

| Метрика эффективности на Test | Модель 1: Логистическая регрессия | Модель 2: Случайный лес (RF) | Модель 3: Градиентный бустинг (CatBoost) |
| :--- | :---: | :---: | :---: |
| **TEST ROC-AUC** | 0.8698 | **0.8724** | 0.8437 |
| **Коэффициент Джини (Gini)** | 73.96% | **74.47%** | 68.74% |
| **Правильно одобрено (True Negative)** | 29 970 | 35 442 | **36 249** |
| **Ложные отказы (False Positive)** | 8 289 | 2 817 | **2 010** |
| **Пропущенные дефолты (False Negative)** | **27** | 63 | 77 |
| **Перехваченные дефолты (True Positive)**| **119** | 83 | 69 |
| **Recall (Полнота класса 1)** | **82%** | 57% | 47% |
| **Precision (Точность класса 1)** | 1% | 3% | **3%** |



### Вывод

1. **Максимальное удержание рисков (Логистическая регрессия):** Данный алгоритм показал наивысшую полноту (`Recall = 82%`), пропустив всего 27 дефолтов. Цена такой безопасности — 8 289 ложных отказов благонадежным клиентам. Модель идеальна для периодов кризиса, когда банку критически важно минимизировать любые финансовые потери.

2. **Оптимальный баланс и качество (Случайный лес):** Модель показала наилучшую обобщающую способность (**Gini = 74.47%**). Она значительно снизила объем ложных отказов (до 2 817) по сравнению с регрессией, сохранив уверенный перехват дефолтов на уровне 57%. Это сбалансированное решение для стабильного рынка.

3. **Максимальная коммерческая эффективность (CatBoost):** Градиентный бустинг оказался самым «одобряющим» алгоритмом. Он снизил ложные отказы до минимума — **2 010 клиентов**, что позволило выдать кредиты максимальному числу надежных заемщиков (**36 249**). Данный подход максимизирует операционную прибыль банка и лояльность клиентов за счет минимизации необоснованных отказов, слегка жертвуя полнотой контроля рисков.




