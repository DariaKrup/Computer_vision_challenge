# Computer_vision_challenge

**Описание**

Предположим что мы с вами разрабатываем инструмент проверки подлинности бумажной банкноты. Для идентификации банкноты используются многоугольники произвольной формы без самопересечений. Необходимо реализовать программу производящую поиск всех контрольных меток на изображении. На вход программа принимает изображение, а также список базисных многоугольников. В ответе необходимо указать количество многоугольников, для каждого многоугольника указать его номер во входном списке, центр и угол поворота.

**Вход**

Число N, а далее на N последующих строках заданы координаты X, Y - отрезков задающих многоугольник. Гарантируется что многоугольники несамопересекающиеся, а также замкнуты.  А также изображение разрешением 300 на 200 пикселей. 

**Запуск программы**

shape_finder.py -s input.txt -i image.png

**Выход**

Число M - количество обнаруженных примитивов, а на следующих M строках - номер фигуры, смещение х, смещение у, масштаб и угол поворота. 
