from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def prepare_data(explanation_set):
    return


def translate(explanation1, explanation2, pred=False):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(explanation1, explanation2, test_size=0.2, random_state=43)
    model.fit(X_train, y_train)
    B = model.coef_
    y_pred = model.predict(X_test)
    scores = model.score(X_test, y_test)
    if pred:
        mse = mean_squared_error(y_test, y_pred)
        return mse, scores
    else:
        return scores
    

def translate_pairwise(explanation_set, pred=True):
    length_explanations = int(len(explanation_set) / 5)
    r2 = {}
    mse = {}
    number_method = {0: 'IG', 1: 'KS', 2: 'LI', 3: 'SG', 4: 'VG'}
    for i in range(5):
        for j in range(5):
            if i != j:
                explanation1 = explanation_set[i*length_explanations:(i+1)*length_explanations].numpy()
                explanation2 = explanation_set[j*length_explanations:(j+1)*length_explanations].numpy()
                mse_score, r2_score = translate(explanation1, explanation2, pred)
                r2[number_method[i] + '_' + number_method[j]] = r2_score
                mse[number_method[i] + '_' + number_method[j]] = mse_score
    return r2, mse



