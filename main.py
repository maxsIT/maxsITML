from notebooks.model_training import train_classifiers
from data_loaders import load_processed_data
import warnings
warnings.filterwarnings('ignore')

# Загружаем данные
fict_df = load_processed_data()

# Оставляем только нужные колонки
data_full = fict_df.drop(
    [
    'name', 
    'surname',
    'is_fict',
    'is_fim',
    'is_gef'
    ], 
    axis=1).copy()
X_data = data_full.drop('is_fem', axis=1)
y = data_full.is_fem

# Проводим исследование моделей
print ("123")
fem_models = train_classifiers(X_data, y)

score_testing_dataset(fem_models[5])
