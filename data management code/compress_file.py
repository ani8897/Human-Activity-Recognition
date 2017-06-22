import csv
import os

file_dir = os.path.dirname(__file__)

User_dict = {'a': -0.4,'b': -0.3,'c': -0.2,
			'd': -0.1,'e': 0.1,'f': 0.2,
			'g': 0.3,'h': 0.4,'i': 0}

Activity_dict = {'stand':1,'sit':2,'walk':3,
				'stairsup':4,'stairsdown':5,'bike':6}

Model_dict = {'nexus4':1, "s3":2, "s3mini":3, "samsungold":4 }

for i in range(1,14):
	print(i)
	file_path= os.path.join(file_dir,'data/pa'+str(i)+'.csv')
	file_path2 = os.path.join(file_dir,'Compressed_data/pa'+str(i)+'.csv')
	with open(file_path) as f:
		read = csv.reader(f)
		data = list(read)

	with open(file_path2,'w') as file:
		csv_file = csv.writer(file)
		#head = ['index','x','y','z','User','Model','Activity']
		#csv_file.writerow(head)
		j=0
		for i in range(0,len(data)):
			if(i%10 == 1):
				data_item = [j]
				'''sumx=0
				sumy=0
				sumz=0
				for k in range(0,10):
					if(i+k < len(data)):
						sumx += float(data[i+k][3])
						sumy += float(data[i+k][4])
						sumz += float(data[i+k][5])'''
				data_item.append(data[i][3])
				data_item.append(data[i][4])
				data_item.append(data[i][5])
				if(data[i][9] != "null"):
					#data_item.append(User_dict[data[i][6]])
					#data_item.append(Model_dict[data[i][7]])
					data_item.append(Activity_dict[data[i][9]])
					csv_file.writerow(data_item)
				j+=1
	




