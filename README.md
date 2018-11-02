# mrg_mlcourse_module1
MNIST digit recognizer
Используется алгоритм softmax на mini-batch gradient descend, размер батча 32.  
В данные добавлены квадратичные и кубические признаки, затем данные отнормированы.
Было проведено порядка 1200 итераций, что заняло всего 10 минут.



Поведение на тренировочных данных:
             precision    recall  f1-score   support

          0       0.98      0.98      0.98      5953
          1       0.98      0.98      0.98      6779
          2       0.93      0.95      0.94      5848
          3       0.93      0.94      0.93      6067
          4       0.95      0.95      0.95      5835
          5       0.91      0.93      0.92      5350
          6       0.97      0.96      0.97      5962
          7       0.96      0.96      0.96      6215
          8       0.93      0.91      0.92      5953
          9       0.94      0.93      0.94      6038

avg / total       0.95      0.95      0.95     60000
Поведение на тестовых данных
             precision    recall  f1-score   support

          0       0.98      0.96      0.97      1005
          1       0.99      0.97      0.98      1160
          2       0.90      0.93      0.92       993
          3       0.92      0.91      0.91      1025
          4       0.94      0.95      0.94       968
          5       0.88      0.90      0.89       869
          6       0.95      0.94      0.95       965
          7       0.92      0.93      0.93      1016
          8       0.89      0.89      0.89       981
          9       0.92      0.92      0.92      1018

avg / total       0.93      0.93      0.93     10000

При увеличении числа эпох качество должно возрасти.
