from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
import pandas as pd
from keras.optimizers import SGD
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
import pickle
from full_pipeline import Stacked_obj


def cmToM(row):
    m = row / 100
    return m


def bmiCalc(height, weight):
    bmi = weight / (height ** 2)
    return bmi


def mapCalc(sys, dias):
    map = (1 / 3) * sys + (2 / 3) * dias
    return map


ITERATION = 0

df = pd.read_csv(
    '/Users/mnemesur/Documents/Dartmouth_QBS/Jacobson_Lab/French_Undergrad_Project/French_Undergrad_Data.csv')
data_convert_dict = {'yes': 1,
                     'no': 0}
df = df.replace(data_convert_dict)

df_dummies = pd.get_dummies(df)
imp_median = IterativeImputer(max_iter=10, random_state=1)
df_imputed = imp_median.fit_transform(df_dummies)
df_imputed = pd.DataFrame(df_imputed)
df_imputed.columns = df_dummies.columns
df_imputed.index = df_dummies.index
df_imputed['Height (cm)'] = cmToM(df_imputed['Height (cm)'])
df_imputed.rename(columns={'Height (cm)': 'Height (m)'}, inplace=True)
df_imputed['BMI'] = bmiCalc(df_imputed['Height (m)'], df_imputed['Weight (kg)'])
df_imputed['pulse_pressure'] = df_imputed['Systolic blood pressure (mmHg)'] - df_imputed[
    'Diastolic blood pressure (mmHg)']
df_imputed['MAP'] = bmiCalc(df_imputed['Systolic blood pressure (mmHg)'],
                            df_imputed['Diastolic blood pressure (mmHg)'])

x = df_imputed.values  # returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_imputed_norm = pd.DataFrame(x_scaled)
df_imputed_norm.columns = df_imputed.columns

x = df_imputed_norm.drop(columns=['Anxiety symptoms', 'Panic attack symptoms', 'Depressive symptoms'])
y = df_imputed_norm['Anxiety symptoms']

class_weight = 5
learning_rate = .020945
decay_rate = 0.00092617
momentum = .97981
nesterov = True
layers = [30, 30, 30, 30, 30]

c_w = {0: 1., 1: class_weight * 1.}
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=nesterov)

classifier = Sequential()
classifier.add(Dense(units=layers[0], activation='relu', input_dim=x.shape[1]))
[classifier.add(Dense(units=layers[i + 1], activation='relu')) for i in range(len(layers) - 1)]
classifier.add(Dense(units=1, activation='sigmoid'))
classifier.compile(optimizer=sgd, loss='binary_crossentropy')
with open('test_object.pkl', 'wb') as output:
    french_obj = Stacked_obj(x, y, [classifier,
                                    xgb.XGBClassifier(n_estimators=500,
                                                      random_state=33),
                                    RandomForestClassifier(n_estimators=500,
                                                           random_state=312),
                                    LogisticRegression(random_state=322),
                                    KNeighborsClassifier(n_neighbors=5),
                                    SVC(kernel='rbf',
                                        #random_state=22,
                                        #probability=True),
                                    # SGDClassifier(random_state=34,
                                    # loss = 'log',
                                    # penalty='elasticnet')
                                    ], upper_mod=xgb.XGBClassifier(random_state=33), regression=False, held_out=True)

    french_obj.train_models(folds=5, folds_upper=5)
    french_obj.display_results_train(save=True, file_name='Anxiety_train_ex.png')
    french_obj.display_results_test(save=True, file_name='Anxiety_test_ex.png')
    pickle.dump(french_obj, output, pickle.HIGHEST_PROTOCOL)
