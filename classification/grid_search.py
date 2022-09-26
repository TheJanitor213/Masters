from subprocess import Popen
import subprocess

gnuplot_exe = r"C:\Program Files (x86)\gnuplot\bin\gnuplot.exe"
grid_py = r"C:\Users\bruce\Documents\Personal-projects\backup\gpu\tools\grid.py"
svmtrain_exe = r"C:\Users\bruce\Documents\Personal-projects\backup\gpu\windows\svm-train-gpu.exe"
svmpredict_exe = r"C:\Users\bruce\Documents\Personal-projects\backup\gpu\windows\svm-predict.exe"
crange = "-5,13,2" #"1,5,2"
grange = "-15,-8,2" #"-3,2,2"
def paramsfromexternalgridsearch(filename, crange, grange, printlines=False):
	#printlines specifies whether or not the function should print every line of the grid search verbosely
	cmd = 'python "{0}" -log2c {1} -log2g {2} -svmtrain "{3}" -gnuplot "{4}" -png grid.png "{5}"'.format(grid_py, crange, grange, svmtrain_exe, gnuplot_exe, filename)
	f = Popen(cmd, shell = True, stdout = subprocess.PIPE).stdout

	line = ''
	while True:
		last_line = line
		line = f.readline()
		if not line: break
		if printlines: print(line)
	c,g,rate = map(float,last_line.split())
	return c,g,rate

print(paramsfromexternalgridsearch("../Accent-Recognition-2019-master/new_training_data",crange, grange, printlines=True))