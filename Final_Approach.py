#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ids = test[['Item_Identifier', 'Outlet_Identifier']]

train['source'] = 'train'
test['source'] = 'test'
data = pd.concat([train, test], ignore_index=True)

print("Data shape:", data.shape)

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
})

item_weight_mean = data.groupby('Item_Identifier')['Item_Weight'].transform('mean')
data['Item_Weight'] = data['Item_Weight'].fillna(item_weight_mean)
data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

outlet_size_mode = data.groupby('Outlet_Type')['Outlet_Size'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Medium')
mode_map = data['Outlet_Type'].map(outlet_size_mode)
data['Outlet_Size'] = data['Outlet_Size'].fillna(mode_map)

data['Item_Visibility'] = data.groupby('Item_Identifier')['Item_Visibility'].transform(
    lambda x: x.replace(0, x.median() if x.median() > 0 else data['Item_Visibility'].median())
)

data['Outlet_Years'] = 2025 - data['Outlet_Establishment_Year']

data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({
    'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'
})

data.loc[data['Item_Type_Combined'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'

data['Price_per_UnitW'] = data['Item_MRP'] / (data['Item_Weight'] + 0.01)

visibility_avg = data.groupby('Item_Identifier')['Item_Visibility'].transform('mean')
data['Item_Visibility_Ratio'] = data['Item_Visibility'] / (visibility_avg + 0.001)

item_popularity = data.groupby('Item_Identifier').size()
data['Item_Popularity'] = data['Item_Identifier'].map(item_popularity)

data['Item_MRP_bin'] = pd.cut(data['Item_MRP'], bins=[0, 69, 136, 203, 400], labels=[0, 1, 2, 3])

data['Visibility_MRP'] = data['Item_Visibility'] * data['Item_MRP']
data['Weight_MRP'] = data['Item_Weight'] * data['Item_MRP']

le = LabelEncoder()
categorical_cols = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 
                   'Outlet_Type', 'Item_Type_Combined']

for col in categorical_cols:
    data[col] = le.fit_transform(data[col].astype(str))

data = pd.get_dummies(data, columns=['Item_Type'], prefix='ItemType')

train_data = data[data['source'] == 'train'].copy()
test_data = data[data['source'] == 'test'].copy()

outlet_item_mean = train_data.groupby(['Outlet_Identifier', 'Item_Type_Combined'])['Item_Outlet_Sales'].mean()
outlet_performance = train_data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean()

train_data['Outlet_Item_Mean'] = train_data.set_index(['Outlet_Identifier', 'Item_Type_Combined']).index.map(outlet_item_mean)
test_data['Outlet_Item_Mean'] = test_data.set_index(['Outlet_Identifier', 'Item_Type_Combined']).index.map(outlet_item_mean)

train_data['Outlet_Performance'] = train_data['Outlet_Identifier'].map(outlet_performance)
test_data['Outlet_Performance'] = test_data['Outlet_Identifier'].map(outlet_performance)

overall_mean = train_data['Item_Outlet_Sales'].mean()
train_data['Outlet_Item_Mean'].fillna(overall_mean, inplace=True)
test_data['Outlet_Item_Mean'].fillna(overall_mean, inplace=True)
train_data['Outlet_Performance'].fillna(overall_mean, inplace=True)
test_data['Outlet_Performance'].fillna(overall_mean, inplace=True)

feature_cols = [
    'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years',
    'Price_per_UnitW', 'Item_Visibility_Ratio', 'Item_Popularity',
    'Visibility_MRP', 'Weight_MRP', 'Item_MRP_bin',
    'Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size',
    'Outlet_Type', 'Item_Type_Combined',
    'Outlet_Item_Mean', 'Outlet_Performance'
]

item_type_cols = [col for col in train_data.columns if col.startswith('ItemType_')]
feature_cols.extend(item_type_cols)

X = train_data[feature_cols].fillna(0)
y = train_data['Item_Outlet_Sales']
X_test = test_data[feature_cols].fillna(0)

print(f"Feature count: {len(feature_cols)}")
print(f"Training set shape: {X.shape}")

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
print(f"Random Forest RMSE: {rf_rmse:.4f}")

print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_preds = gb_model.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_preds))
print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")

print("Creating ensemble...")
ensemble_preds = 0.6 * gb_preds + 0.4 * rf_preds
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_preds))
print(f"Ensemble RMSE: {ensemble_rmse:.4f}")

best_rmse = min(rf_rmse, gb_rmse, ensemble_rmse)
print(f"\nBest validation RMSE: {best_rmse:.4f}")

if ensemble_rmse == best_rmse:
    print("Using ensemble for final predictions")
    final_preds = 0.6 * gb_model.predict(X_test) + 0.4 * rf_model.predict(X_test)
elif gb_rmse < rf_rmse:
    print("Using Gradient Boosting for final predictions")
    final_preds = gb_model.predict(X_test)
else:
    print("Using Random Forest for final predictions")
    final_preds = rf_model.predict(X_test)

submission = test_ids.copy()
submission['Item_Outlet_Sales'] = final_preds
submission.to_csv('BigMart_Simple_Enhanced_Submission.csv', index=False)

print("Submission created: BigMart_Simple_Enhanced_Submission.csv")
print(f"Expected score improvement: {best_rmse:.0f} (from your current 1156)")

if gb_rmse <= rf_rmse:
    importance_model = gb_model
    importance_values = gb_model.feature_importances_
else:
    importance_model = rf_model
    importance_values = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance_values
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))