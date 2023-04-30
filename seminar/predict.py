import mlflow
logged_model = 'runs:/1d8358022de9406aba2e760e86cdd5d7/lr_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = {
    'Col1' : [1,6], 'Col2':[1, 8], 'Col3' : [1, 65],
    'Col4' : [1,2], 'Col5':[1, 25], 'Col6' : [1, 3],
    'Col7' : [1,22], 'Col8':[1, 2], 'Col9' : [1, 6],
    'Col10' : [1,6], 'Col11':[1, 2], 'Col12' : [1, 5],
    'Col13' : [1,1], 'Col14':[1, 2], 'Col15' : [1, 1],
    'Col16' : [1,5], 'Col17':[1, 6], 'Col18' : [1, 2387],
    'Col19' : [1,3], 'Col20':[1, 23], 'Col21' : [1, 655],
    'Col22' : [1,1], 'Col23':[1, 268], 'Col24' : [1, 3],
    'Col25' : [1,12], 'Col26':[1, 24], 'Col27' : [1, 6],
    'Col28' : [1,0], 'Col29':[1, 2056]
}

print(loaded_model.predict(pd.DataFrame(data)))