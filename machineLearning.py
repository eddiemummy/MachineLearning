def get_model(X,y,test_size,models=None):
    if not isinstance(models,list):
        raise TypeError("Model must be in a list format")
    # split data into train, validation and test
    x_new,x_test,y_new,y_test = train_test_split(X,y,test_size=test_size,random_state=0)
    x_train,x_dev,y_train,y_dev = train_test_split(x_new,y_new,test_size=test_size,random_state=0)
    ####
    x_s = [x_train,x_dev,x_test]
    y_s = [y_train,y_dev,y_test]
    names =  ["train","dev","test"]
    models = models
    scores = []
    for m in models:
        for i in range(3):
            model = m.fit(x_s[0],y_s[0])
            pred = model.predict(x_s[i])
            acc = accuracy_score(pred,y_s[i])
            scores.append(acc)
    import pandas as pd
    import numpy as np
    columns = ["train","validation","test"]
    index = [m.__class__.__name__ for m in models]
    shape_col = int(len(scores)/3)
    data = np.array(scores).reshape(shape_col,3)    
    df = pd.DataFrame(data=data,index=index,columns=columns)
    return df.transpose()