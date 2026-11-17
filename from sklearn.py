from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# ---------------- KNN ----------------
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_test_acc = accuracy_score(y_test, knn_pred)

knn_cv_acc = cross_val_score(knn, X, y, cv=5).mean()

print("KNN Test Accuracy:", knn_test_acc)
print("KNN CV Accuracy:", knn_cv_acc)

# ---------------- Decision Tree ----------------
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_test_acc = accuracy_score(y_test, dt_pred)

dt_cv_acc = cross_val_score(dt, X, y, cv=5).mean()

print("Decision Tree Test Accuracy:", dt_test_acc)
print("Decision Tree CV Accuracy:", dt_cv_acc)

# ---------------- Random Forest ----------------
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_test_acc = accuracy_score(y_test, rf_pred)

rf_cv_acc = cross_val_score(rf, X, y, cv=5).mean()

print("Random Forest Test Accuracy:", rf_test_acc)
print("Random Forest CV Accuracy:", rf_cv_acc)

bjnbnjb