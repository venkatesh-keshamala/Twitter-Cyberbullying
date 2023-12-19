def test_user_data(data):
    X_train, X_test, y_train, y_test, clf = fitting(X, y, 'gini', 'best', mdepth=3, clweight=None, minleaf=1000)
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict([data])
    return y_pred