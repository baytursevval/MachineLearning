import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/sevval/Desktop/python/machine/multi_linear_regression_dataset.csv")


deneyim_yas=data.loc[:, ['deneyim','yas']].values
maas=data['maas'].values.reshape(-1,1) #array şekline dönüştürür

#algorithm
import sklearn.linear_model as ln 
reg=ln.LinearRegression() 

#data split
import sklearn.model_selection as ms
#ms: yardımcı fonk. datanın test ve train olarak 2ye bölmeyi sağlar
x_train, x_test, y_train, y_test=ms.train_test_split(deneyim_yas,maas, test_size=1/3, random_state=0)
#random_state: modeli her çalıştırdığında farklı farklı seçilimler yapmasındansa 1 kere bir
# random seçilim yapıp sonrasında o seçilim ile devam etmesini sağlar


#train
reg.fit(x_train, y_train)

#predict
y_pred=reg.predict(x_test)

print("deneyim ve yaslar: ",x_test)
print("tahmin edilen maaslar:", y_pred)

#score
import sklearn.metrics as mt
score=mt.r2_score(y_test,y_pred)
print("score:",score)

#graph
plt.scatter(deneyim_yas[:,0],maas,color="r")
plt.scatter(x_test[:,0],y_pred,color="b")
plt.show()


def maasTahminEt(deneyim):
    return reg.predict(deneyim)

while True:
    deneyim_giris=input("deneyim gir:")
    maas_tahmini=maasTahminEt([ [int(deneyim_giris)]])
    print("verilmesi gereken maas: ", maas_tahmini)