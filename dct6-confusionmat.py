import numpy as np
from sklearn.metrics import confusion_matrix
# A: 아델리
# C: 친스트랩
y_true=np.array(["A",'A','C',"A",'C','C','C'])
y_pred=np.array(["A",'C','A',"A",'A','C','C'])
y_pred2=np.array(["C",'A','A',"A",'C','C','C'])

conf_mat=confusion_matrix(y_true=y_true, 
                 y_pred=y_pred,              labels=["A","C"])

conf_mat
from sklearn.metrics import ConfusionMatrixDisplay
p=ConfusionMatrixDisplay(confusion_matrix=conf_mat, 
                         display_labels=("Adelie", "Chinstrap"))
p.plot(cmap="Blues")

##

conf_mat2=confusion_matrix(y_true=y_true, 
                 y_pred=y_pred2, 
                 labels=["A","C"])

from sklearn.metrics import ConfusionMatrixDisplay
p=ConfusionMatrixDisplay(confusion_matrix=conf_mat2, 
                         display_labels=("Adelie", "Chinstrap"))
p.plot(cmap="Greens")