import sys
sys.path.append('../..')
import glob
from bs4 import BeautifulSoup
import json
import sys
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import laser_lib as ll
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D


def create_number_marker(number, size=1.0):
    text_path = TextPath((0, 0), str(number), size=size)
    trans = Affine2D().translate(-text_path.vertices[:, 0].mean(), -text_path.vertices[:, 1].mean())
    text_path = trans.transform_path(text_path)
    return text_path



def calculate_median(lst):
    sorted_lst = sorted(lst)
    n = len(sorted_lst)
    mid = n // 2  # Floor division to get the middle index

    if n % 2 == 0:
        # If the list has an even number of elements, average the two middle values
        median = (sorted_lst[mid - 1] + sorted_lst[mid]) / 2
    else:
        # If the list has an odd number of elements, the median is the middle value
        median = sorted_lst[mid]

    
    
    return [sorted_lst[0], median, sorted_lst[-1]]


def calc_bank_errors(bank_row_dict, bank):
	errors = 0
	for row in bank_row_dict[bank]:
		for col in bank_row_dict[bank][row]:
			for bit in bank_row_dict[bank][row][col]:
				errors +=  bank_row_dict[bank][row][col][bit][0] + bank_row_dict[bank][row][col][bit][1]
	return errors
	

lines = ll.parse_run_directory(sys.argv[1], sys.argv[2])


### DDR4 Parameter, set according too need.
if((len(sys.argv)>3) and (sys.argv[3]=="DDR4")):
	DDR4 =True
else:
	DDR4 = False


#######################################################################################################################################################################################
######################                                         		Mapping the banks spatially 									###############
#######################################################################################################################################################################################



'''
To find the most two/four impacted banks for vertical/horizontal lines respectively, the method used here for each line is to divide the number of errors 
for a specific bank by the maximum errors found for that specifc bank by each line. This ratio gives a value ( [0,1)) for each bank impacted by that specific 
line. Using this number we select the most impacted two/four banks for vertical/horizontal lines respectively. 
'''
max_bank_errors = {}
for line in lines :
	for bank in line.bank_row_dict:
		if bank not in max_bank_errors:
			max_bank_errors[bank] = 0
		if calc_bank_errors(line.bank_row_dict, bank) > max_bank_errors[bank]:
			max_bank_errors[bank] = calc_bank_errors(line.bank_row_dict, bank)



for line in lines:
	bank_errors_scaled = {}
	for bank in line.bank_row_dict:
		bank_errors_scaled[bank] = calc_bank_errors(line.bank_row_dict, bank) / max_bank_errors[bank]
	bank_errors_scaled_sorted = sorted(bank_errors_scaled.items(), key=lambda item: item[1], reverse=True)
	print(line.get_line_name())
	print(bank_errors_scaled_sorted)
	if line.coor == "X":
		if len(bank_errors_scaled_sorted)>0:
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[0][0]] = line.bank_row_dict[bank_errors_scaled_sorted[0][0]]
		if len(bank_errors_scaled_sorted)>1:
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[1][0]] = line.bank_row_dict[bank_errors_scaled_sorted[1][0]]	
		if DDR4:
			if len(bank_errors_scaled_sorted)>2:						
				line.bank_row_dict_max_banks[bank_errors_scaled_sorted[2][0]] = line.bank_row_dict[bank_errors_scaled_sorted[2][0]]
			if len(bank_errors_scaled_sorted)>3:	
				line.bank_row_dict_max_banks[bank_errors_scaled_sorted[3][0]] = line.bank_row_dict[bank_errors_scaled_sorted[3][0]]	
			
	else:
		if len(bank_errors_scaled_sorted)>0:	
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[0][0]] = line.bank_row_dict[bank_errors_scaled_sorted[0][0]]
		if len(bank_errors_scaled_sorted)>1:		
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[1][0]] = line.bank_row_dict[bank_errors_scaled_sorted[1][0]]
		if len(bank_errors_scaled_sorted)>2:						
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[2][0]] = line.bank_row_dict[bank_errors_scaled_sorted[2][0]]
		if len(bank_errors_scaled_sorted)>3:	
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[3][0]] = line.bank_row_dict[bank_errors_scaled_sorted[3][0]]		



lines = sorted(lines, key=lambda line: line.name)

for line in lines:
	if(line.coor == "X") or (line.coor == "Y"):
		print("***********************"+line.coor + " = " +str(line.value)+"*******************************")
		for bank in line.bank_row_dict_max_banks:
			print(bank, calc_bank_errors(line.bank_row_dict_max_banks, bank))


	



#### Do X,Y Correlation on this bank data to plot spatially.
banks = {}
for line in lines:
	for bank in line.bank_row_dict_max_banks:
		if bank not in banks:
			banks[bank] = [[],[]]
		if line.coor == "X":
			banks[bank][0].append(line.value)
		if line.coor == "Y":
			banks[bank][1].append(line.value)
banks2 = {}
for bank in banks:
	banks2[bank] = []
	for X in banks[bank][0]:
		for Y in banks[bank][1]:
			banks2[bank].append([X,Y])

## Plot the banks
banks_list =  list(banks2.keys())



print(banks_list)
scatter_plot =[]


for bank in banks2:
	bank_id = banks_list.index(bank)
	X =[]
	Y = []
	bank_colors = []
	for point in banks2[bank]:
		X.append(point[0])
		Y.append(point[1])
	

	print(X)
	print(Y)
	print(bank_colors)
	print(bank_id)
	scatter_plot.append(plt.scatter(X, Y, color='black', s=0, marker="o" )	)
	for x,y in zip(X, Y):
		plt.text(x,y, str(bank), color="red", fontsize=6)	
plt.title("Spatial Mapping of Banks")
plt.xlim(0,110)
plt.ylim(0,110)
#legend1 = plt.legend(scatter_plot, banks2.keys(), loc="upper left")
#plt.gca().add_artist(legend1)  # Add the color-based legend to the plot without removing the scatter plot legend

plt.show()
plt.close()



bank_overlap = np.zeros((110, 110))
for bank in banks2:
	for coor in banks2[bank]:
		bank_overlap[coor[0]][coor[1]] +=0.0625


plt.imshow(bank_overlap, cmap='Greys', interpolation='nearest', alpha=0.2)
plt.title('Bank Overlap')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()
plt.close()



######################################################################################################
################# Calculate Spatial Correlation of Rows and Columns ##################################
######################################################################################################


##################### Calculate XY Correlation data #####################


rows = {}

for line in lines:
	for bank in line.bank_row_dict_max_banks:
		for row in line.bank_row_dict_max_banks[bank]:
			key = str(bank)+"_"+str(row)
			if key not in rows:
				rows[key] = [[],[]]
			if line.coor == "X":
				rows[key][0].append(line.value)
			if line.coor == "Y":
				rows[key][1].append(line.value)
rows2 = {}
for key in rows:
	for X in rows[key][0]:
		for Y in rows[key][1]:
			if key not in rows2:
				rows2[key] = []				
			rows2[key].append([X,Y])



cols = {}

for line in lines:
	for bank in line.bank_row_dict_max_banks:
		for row in line.bank_row_dict_max_banks[bank]:
			for col in line.bank_row_dict_max_banks[bank][row]:
				key = str(bank)+"_"+str(row)+"_"+str(col)
				if key not in cols:
					cols[key] = [[],[]]
				if line.coor == "X":
					cols[key][0].append(line.value)
				if line.coor == "Y":
					cols[key][1].append(line.value)
cols2 = {}
for key in cols:
	for X in cols[key][0]:
		for Y in cols[key][1]:
			if key not in cols2:
				cols2[key] = []	
			cols2[key].append([X,Y])





#######################################################################################################################################################################################
######################                                         Spatial Correlation for Rows and Columns in a single direction 						###############
#######################################################################################################################################################################################


row_correlation_x = {}
row_correlation_x_data = []
row_correlation_y = {}
row_correlation_y_data = []
col_correlation_x = {}
col_correlation_x_data = []
col_correlation_y = {}
col_correlation_y_data = []
bit_correlation_x = {}
bit_correlation_x_data = []
bit_correlation_y = {}
bit_correlation_y_data =[]
#Correlation between lines:
#X_correlations:	
print("=====================================X_Cols_data========================================")
for line1 in lines:
	for line2 in lines:
		if line1.coor == "X" and line2.coor == "X":
			diff = line1.value - line2.value
			if diff > 0:
				if diff not in row_correlation_x:
					row_correlation_x[diff] = 0
				if diff not in col_correlation_x:
					col_correlation_x[diff] = 0
				if diff not in bit_correlation_x:
					bit_correlation_x[diff] = 0
				
				for bank in line1.bank_row_dict_max_banks:
					if bank in line2.bank_row_dict_max_banks:
						for row in line1.bank_row_dict_max_banks[bank]:
							if row in line2.bank_row_dict_max_banks[bank]:
								print(line1.value,bank, row)
								print(line1.bank_row_dict_max_banks[bank][row].keys())
								print(line2.value,bank, row)
								print(line2.bank_row_dict_max_banks[bank][row].keys())

								row_correlation_x[diff] += 1
								row_correlation_x_data.append([bank,row,line1.value, line2.value])
								for col in line1.bank_row_dict_max_banks[bank][row]:
									if col in line2.bank_row_dict_max_banks[bank][row]:
										col_correlation_x[diff] += 1
										col_correlation_x_data.append([bank,row,col,line1.value, line2.value])
										for bit in line1.bank_row_dict_max_banks[bank][row][col]:
											if bit in line2.bank_row_dict_max_banks[bank][row][col]:
												bit_correlation_x[diff] += 1
												bit_correlation_x_data.append([bank,row,col,bit,line1.value, line2.value])

#Y_correlations:	
print("=====================================Y_Cols_data=====================================")
for line1 in lines:
	for line2 in lines:
		if line1.coor == "Y" and line2.coor == "Y":
			diff = line1.value - line2.value
			if diff > 0:
				if diff not in row_correlation_y:
					row_correlation_y[diff] = 0
				if diff not in col_correlation_y:
					col_correlation_y[diff] = 0
				if diff not in bit_correlation_y:
					bit_correlation_y[diff] = 0
				
				for bank in line1.bank_row_dict_max_banks:
					if bank in line2.bank_row_dict_max_banks:
						for row in line1.bank_row_dict_max_banks[bank]:
							if row in line2.bank_row_dict_max_banks[bank]:
								print(line1.value,bank, row)
								print(line1.bank_row_dict_max_banks[bank][row].keys())
								print(line2.value,bank, row)
								print(line2.bank_row_dict_max_banks[bank][row].keys())
								row_correlation_y[diff] += 1
								row_correlation_y_data.append([bank,row,line1.value, line2.value])
								for col in line1.bank_row_dict_max_banks[bank][row]:
									if col in line2.bank_row_dict_max_banks[bank][row]:
										col_correlation_y[diff] += 1
										col_correlation_y_data.append([bank,row,col,line1.value, line2.value])
										for bit in line1.bank_row_dict_max_banks[bank][row][col]:
											if bit in line2.bank_row_dict_max_banks[bank][row][col]:
												bit_correlation_y[diff] += 1
												bit_correlation_y_data.append([bank,row,col,bit,line1.value, line2.value])










#print(row_correlation_x)
#print(row_correlation_y)
#print(col_correlation_x)
#print(col_correlation_y)
#print(bit_correlation_x)
#print(bit_correlation_y)

fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # 2 rows, 1 column


X = []
Y = []

for diff in row_correlation_x: 
	
	X.append(diff)
	Y.append(row_correlation_x[diff])

sorted_data = sorted(zip(X, Y))
X_sorted, Y_sorted = zip(*sorted_data)
print(X_sorted)
print(Y_sorted)
axes[0,0].plot(X_sorted,Y_sorted)		
axes[0,0].set_title("Row Correlation X")

X = []
Y = []

for diff in col_correlation_x: 
	X.append(diff)
	Y.append(col_correlation_x[diff])

sorted_data = sorted(zip(X, Y))
X_sorted, Y_sorted = zip(*sorted_data)
print(X_sorted)
print(Y_sorted)
axes[0,1].plot(X_sorted,Y_sorted)		
axes[0,1].set_title("Col Correlation X")


X = []
Y = []


for diff in bit_correlation_x: 
	X.append(diff)
	Y.append(bit_correlation_x[diff])

sorted_data = sorted(zip(X, Y))
X_sorted, Y_sorted = zip(*sorted_data)
print(X_sorted)
print(Y_sorted)
axes[0,2].plot(X_sorted,Y_sorted)		
axes[0,2].set_title("Bit Correlation X")



X = []
Y = []

for diff in row_correlation_y:
	X.append(diff)
	Y.append(row_correlation_y[diff])

sorted_data = sorted(zip(X, Y))
X_sorted, Y_sorted = zip(*sorted_data)

print(X_sorted)
print(Y_sorted)
axes[1,0].plot(X_sorted,Y_sorted)		
axes[1,0].set_title("Row Correlation Y")



X = []
Y = []

for diff in col_correlation_y: 
	X.append(diff)
	Y.append(col_correlation_y[diff])

sorted_data = sorted(zip(X, Y))
X_sorted, Y_sorted = zip(*sorted_data)
print(X_sorted)
print(Y_sorted)
axes[1,1].plot(X_sorted,Y_sorted)		
axes[1,1].set_title("Col Correlation Y")


X = []
Y = []

for diff in bit_correlation_y: 
	X.append(diff)
	Y.append(bit_correlation_y[diff])

sorted_data = sorted(zip(X, Y))
X_sorted, Y_sorted = zip(*sorted_data)
print(X_sorted)
print(Y_sorted)
axes[1,2].plot(X_sorted,Y_sorted)		
axes[1,2].set_title("Bit Correlation Y")

plt.show()




######################################################################################################################################
#########################  Intended for finer mapping plotting the spread of rows bank wise ##########################################
######################################################################################################################################
'''
X_spread= [] 
Y_spread= []
X_min = []
X_max = []
Y_min = []
Y_max = []
xt = 2

for line in lines:
	bank_coff =0
	for bank in line.bank_row_dict_max_banks:
		rows = sorted(list(line.bank_row_dict_max_banks[bank].keys()));
		rows_based = []
		for row in rows:
			rows_based.append(row - rows[0]);


        
		if(line.coor=="X"):
			plt.scatter([line.value*10 + bank_coff]*len(rows_based), rows_based, s=2)
		bank_coff +=5
		


plt.title("Spread of rows in lines drawn parallel to Y Axis.")
plt.show()
	
plt.close()


X_spread= [] 
Y_spread= []
X_min = []
X_max = []
Y_min = []
Y_max = []
xt = 2
for line in lines:
	bank_coff =0
	for bank in line.bank_row_dict_max_banks:
		rows = sorted(list(line.bank_row_dict_max_banks[bank].keys()));
		rows_based = []
		for row in rows:
			rows_based.append(row - rows[0]);


        
		if(line.coor=="Y"):
			plt.scatter([line.value*20 + bank_coff]*len(rows_based), rows_based, s=2)
		bank_coff+= 3


plt.title("Spread of rows in lines drawn parallel to X Axis")
plt.show()
		

plt.close()
'''
##################### Plot bank based spread of rows in X and Y direction ##############

X_spread= [] 
Y_spread= []
X_min = []
X_max = []
Y_min = []
Y_max = []
xt = 2
if DDR4:
	fig, axes = plt.subplots(4, 4, figsize=(20, 6))  # 2 rows, 1 column
else:
	fig, axes = plt.subplots(2, 4, figsize=(20, 6)) 	

for bank1 in banks_list:
	for line in lines:
		if line.coor == "X":
			bank_coff =0
			for bank in line.bank_row_dict_max_banks:
				if bank1 == bank:
					bank_id = banks_list.index(bank) 
					rows = sorted(list(line.bank_row_dict_max_banks[bank].keys()));
					row_based =[]
					for row in rows:
						row_based.append(row - rows[0]);
					
					axes[int(bank_id/4), bank_id%4].scatter([line.value]*len(row_based), row_based, s=2)
	
#		axes[int(bank_id/4), bank_id%4].scatter(X[i], Y[i], marker=bank_colors[i], s=4, facecolor ="none",  label=str(bank), edgecolor = "black", alpha=1)	


for bank in banks_list:	
	bank_id = banks_list.index(bank) 
	axes[int(bank_id/4), bank_id%4].set_title("bank=" + str(bank))
	#axes[int(bank_id/4), bank_id%4].set_xlim(0,110)
	axes[int(bank_id/4), bank_id%4].set_ylim(0,65536)

	
plt.show()
plt.close()
					
	

X_spread= [] 
Y_spread= []
X_min = []
X_max = []
Y_min = []
Y_max = []
xt = 2
if DDR4:
	fig, axes = plt.subplots(4, 4, figsize=(20, 6))
else:
	fig, axes = plt.subplots(2, 4, figsize=(20, 6))

for bank1 in banks_list:
	for line in lines:
		if(line.coor =="Y"):
			bank_coff =0
			for bank in line.bank_row_dict_max_banks:
				if bank1 == bank:
					bank_id = banks_list.index(bank) 
					rows = sorted(list(line.bank_row_dict_max_banks[bank].keys()));
					row_based =[]
					for row in rows:
						row_based.append(row - rows[0]);
					axes[int(bank_id/4), bank_id%4].scatter([line.value]*len(row_based), row_based, s=2)

#		axes[int(bank_id/4), bank_id%4].scatter(X[i], Y[i], marker=bank_colors[i], s=4, facecolor ="none",  label=str(bank), edgecolor = "black", alpha=1)	


for bank in banks_list:	
	bank_id = banks_list.index(bank) 
	axes[int(bank_id/4), bank_id%4].set_title("bank=" + str(bank))
	#axes[int(bank_id/4), bank_id%4].set_xlim(0,110)
	axes[int(bank_id/4), bank_id%4].set_ylim(0,65536)

	
plt.show()
plt.close()       
	





########################################################################################################################################################################################
############################################# Plot the DR/DE, Jumps in the row numbers for each line and each bank when rows are arranged in ascending order############################
########################################################################################################################################################################################
print("Counter Prints")
labels = []
for bank1 in banks_list:
	bank_lst_x = {}
	for line in lines:
		if(line.coor == "X"):
			for bank in line.bank_row_dict_max_banks:
				if bank1 == bank:
					rows = list(line.bank_row_dict_max_banks[bank].keys())
					rows_sorted = sorted(rows)
					diff = []
					for i in range(len(rows_sorted)-1):
						diff.append(rows_sorted[i+1] - rows_sorted[i])
					labels.append("Bank="+str(bank1)+"  X="+str(line.value))
					print(Counter(diff))
					bank_lst_x[line.value] = [rows_sorted[:-1], diff]

	bank_lst_y = {}
	for line in lines:
		if(line.coor == "Y"):
			for bank in line.bank_row_dict_max_banks:
				if bank1 == bank:
					rows = list(line.bank_row_dict_max_banks[bank].keys())
					rows_sorted = sorted(rows)
					diff = []
					for i in range(len(rows_sorted)-1):
						diff.append(rows_sorted[i+1] - rows_sorted[i])
					labels.append("Bank="+str(bank1)+ "  Y="+str(line.value))
					print(Counter(diff))
					bank_lst_y[line.value] = [rows_sorted[:-1], diff]
	fig, axes = plt.subplots(2, max(len(bank_lst_y), len(bank_lst_x)), figsize=(30, 6))  # 2 rows, 1 column	
	it = 0
	sorted_keys = sorted(bank_lst_x.keys())
	#print(sorted_keys)
	for value in sorted_keys:			
		axes[0, it].plot(bank_lst_x[value][0], bank_lst_x[value][1], marker="o")	
		axes[0, it].set_title("X="+str(value))
		it = it+1
	it = 0
	sorted_keys = sorted(bank_lst_y.keys())
	#print(sorted_keys)
	for value in sorted_keys:			
		axes[1, it].plot(bank_lst_y[value][0], bank_lst_y[value][1], marker="o")
		axes[1, it].set_title("Y="+str(value))
		it = it+1
	fig.suptitle("Bank=" + str(bank1))
	plt.show()
	plt.close()


print(labels)





