from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_model=BaggingClassifier(DecisionTreeClassifier(),
    n_estimators=50,  # bagging에 사용될 모델 개수
    max_samples=100,  #dataset만들때 뽑을 표본의 크기
    n_jobs=-1, 
    random_state=42)

bagging_model.fit(x_train, y_train)

