from sklearn.cross_decomposition import PLSRegression


class PLSRegressionWrapper(PLSRegression):
    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X)
