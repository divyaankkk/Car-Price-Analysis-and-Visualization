data = pd.read_csv("cars.csv")

data = data.dropna()

label_encoder = LabelEncoder()
categorical_cols = ['manufacturer_name', 'model_name', 'transmission',
'color', 'engine_fuel', 'engine_type', 'body_type', 'has_warranty',
'state', 'drivetrain', 'location_region']
for col in categorical_cols:
 data[col] = label_encoder.fit_transform(data[col])
# Remove Columns feature_0 to feature_9
data = data.drop(columns=['feature_0', 'feature_1', 'feature_2',
'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7',
'feature_8', 'feature_9'])


X = data.drop(columns=['price_usd'])
y = data['price_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
y_pred = regression_model.predict(X_test)
print("Multiple Linear Regression Metrics:")
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

X_cls = data.drop(columns=['is_exchangeable'])
y_cls = data['is_exchangeable']
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
classification_model = LogisticRegression()
classification_model.fit(X_train_cls, y_train_cls)
y_pred_cls = classification_model.predict(X_test_cls)
print("\nClassification Metrics:")
print("Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("Classification Report:")
print(classification_report(y_test_cls, y_pred_cls))
print("Confusion Matrix:")
print(confusion_matrix(y_test_cls, y_pred_cls))

X_cluster = data[['odometer_value', 'year_produced', 'price_usd']]
kmeans = KMeans(n_clusters=3)
kmeans.fit(X_cluster)
data['cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='odometer_value', y='price_usd',
hue='cluster', palette='viridis')
plt.title('Odometer Value vs Price (Clustered)')
plt.xlabel('Odometer Value')
plt.ylabel('Price (USD)')
plt.legend(title='Cluster')
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='price_usd', bins=20, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='transmission', y='price_usd')
plt.title('Transmission Type vs Price')
plt.xlabel('Transmission Type')
plt.ylabel('Price (USD)')
plt.xticks([0, 1], ['Automatic', 'Manual']) # Add custom labels
plt.show()

data['transmission'] = label_encoder.fit_transform(data['transmission'])
# Scatter plot: Year produced vs Price
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(data=data, x='year_produced', y='price_usd',
hue='transmission', palette={0:'orange', 1:'blue'})
plt.title('Year Produced vs Price (Colored by Transmission)')
plt.xlabel('Year Produced')
plt.ylabel('Price (USD)')
legend_labels = ['Automatic', 'Manual']
handles, _ = scatter.get_legend_handles_labels()
plt.legend(handles, legend_labels, title='Transmission') # Add legend
plt.show()