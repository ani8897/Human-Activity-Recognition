import csv
import os

file_dir = os.path.dirname(__file__)

file_path2 = os.path.join(file_dir,'adata.csv')

with open(file_path2,'w') as file:
	csv_file = csv.writer(file)
	head = ['index','x','y','z','Activity']
	csv_file.writerow(head)
	for i in range(1,14):
		print(i)
		file_path= os.path.join(file_dir,'pa'+str(i)+'.csv')
		with open(file_path) as f:
			read = csv.reader(f)
			data = list(read)
		for j in range(0,len(data)):
			csv_file.writerow(data[j])
