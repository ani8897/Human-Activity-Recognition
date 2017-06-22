import csv
import os

file_dir = os.path.dirname(__file__)

Activity_dict = {'stand':1,'sit':2,'walk':3,
				'stairsup':4,'stairsdown':5,'bike':6}


for i in range(1,9):
	print(i)
	file_patha = os.path.join(file_dir,'data/pa'+str(i)+'.csv')
	file_pathg = os.path.join(file_dir,'data/pg'+str(i)+'.csv')
	file_path  = os.path.join(file_dir,'Compressed_data/p'+str(i)+'.csv')
	
	with open(file_patha) as fa:
		reada = csv.reader(fa)
		dataa = list(reada)

	with open(file_pathg) as fg:
		readg = csv.reader(fg)
		datag = list(readg)

	with open(file_path,'w') as file:
		csv_file = csv.writer(file)
		countera,counterg,l = 0,0,min(len(datag), len(dataa))
		while(countera < l and counterg < l):
			if(countera%100000 < 10 ):
				print(countera)
			if(dataa[countera][0] == datag[counterg][0]):
				data_item = [countera]
				data_item.append(dataa[i][3])
				data_item.append(dataa[i][4])
				data_item.append(dataa[i][5])
				data_item.append(datag[i][3])
				data_item.append(datag[i][4])
				data_item.append(datag[i][5])
				if(dataa[countera][9] != "null" and datag[counterg][9] != "null"):
					data_item.append(Activity_dict[dataa[i][9]])
					csv_file.writerow(data_item)
					countera+=10
					counterg+=10
				countera+=1
				counterg+=1
			else:
				if(int(dataa[countera][0])<10 and int(datag[counterg][0])>=10):
					counterg = counterg + 10
				elif(int(datag[counterg][0])<10 and int(dataa[countera][0])>=10):
					countera = countera + 10
				elif(int(datag[counterg][0])<int(dataa[countera][0])):
					counterg+=1
				elif(int(dataa[countera][0])<int(datag[counterg][0])):
					countera+=1
