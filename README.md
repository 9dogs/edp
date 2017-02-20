# Hey!
This is very narrow-specialized software suitable for photon echo data 
processing for custom experimental setup in Institute of Spectroscopy of Russian
Academy of Sciences. You possibly don't need it (maybe you can use it as 
example of QtWidgets, Matplotlib, pyqtgraph and lmfit).

## If you still want to run this

### Windows
Best way is to use [Miniconda with Python 3.5](https://conda.io/miniconda.html).
Create new virtual environment:

 ```
 conda create --name edp --file conda-req.txt
 activate edp
 ```
 
 You still need to install packages that is not in conda repos:
 
 ```
 pip install lmfit
 ```
 
 Then run `main.py`: 
 ```
 python main.py
 ```