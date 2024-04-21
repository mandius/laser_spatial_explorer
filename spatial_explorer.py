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

def calc_bank_errors(bank_row_dict, bank):
	errors = 0
	for row in bank_row_dict[bank]:
		for col in bank_row_dict[bank][row]:
			for bit in bank_row_dict[bank][row][col]:
				errors +=  bank_row_dict[bank][row][col][bit][0] + bank_row_dict[bank][row][col][bit][1]
	return errors
	

lines = ll.parse_run_directory(sys.argv[1], sys.argv[2])



#######################################################################################################################################################################################
######################                                         		Mapping the banks spatially 									###############
#######################################################################################################################################################################################




######################## Find the maximum number of errors for each bank################################
max_bank_errors = {}
for line in lines :
	for bank in line.bank_row_dict:
		if bank not in max_bank_errors:
			max_bank_errors[bank] = 0
		if calc_bank_errors(line.bank_row_dict, bank) > max_bank_errors[bank]:
			max_bank_errors[bank] = calc_bank_errors(line.bank_row_dict, bank)


#################### Scale the errors and find the top two banks ###################v
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
	else:
		if len(bank_errors_scaled_sorted)>0:	
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[0][0]] = line.bank_row_dict[bank_errors_scaled_sorted[0][0]]
		if len(bank_errors_scaled_sorted)>1:		
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[1][0]] = line.bank_row_dict[bank_errors_scaled_sorted[1][0]]
		if len(bank_errors_scaled_sorted)>2:						
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[2][0]] = line.bank_row_dict[bank_errors_scaled_sorted[2][0]]
		if len(bank_errors_scaled_sorted)>3:	
			line.bank_row_dict_max_banks[bank_errors_scaled_sorted[3][0]] = line.bank_row_dict[bank_errors_scaled_sorted[3][0]]		

##################### Calculate XY Correlation data #####################

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

banks_list =  list(banks2.keys())


#Separate subplotting for debugging
# To plot the data differentially 
## Choose a colormap
#marker_styles = ['o', 's', '^', 'v', 'D', 'P', '*', 'X']
#fig, axes = plt.subplots(2, 4, figsize=(20, 6))  # 2 rows, 1 column
#
#print(banks_list)
#for bank in banks2:
#	bank_id = banks_list.index(bank)
#	X =[]
#	Y = []
#	bank_colors = []
#	for point in banks2[bank]:
#		X.append(point[0])
#		Y.append(point[1])
#		bank_colors.append(marker_styles[bank_id])
#
#	print(X)
#	print(Y)
#	print(bank_colors)
#	print(bank_id)
#	for i in range(len(X)):
#		axes[int(bank_id/4), bank_id%4].scatter(X[i], Y[i], marker=bank_colors[i], s=4, facecolor ="none",  label=str(bank), edgecolor = "black", alpha=1)		
#		axes[int(bank_id/4), bank_id%4].set_title("bank=" + str(bank))
#		axes[int(bank_id/4), bank_id%4].set_xlim(0,110)
#		axes[int(bank_id/4), bank_id%4].set_ylim(0,80)
#
#		
#
#plt.show()
#plt.close()

## Choose a colormap
marker_colors = ['blue', 'orange', 'green', 'red', 'purple', 'black', 'pink', 'gray']


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
	bank_color = marker_colors[bank_id]

	print(X)
	print(Y)
	print(bank_colors)
	print(bank_id)
	
	scatter_plot.append(plt.scatter(X, Y, color=bank_color, s=30,   label=str(bank), edgecolor = "black", alpha=1)	)	
plt.title("Spatial Mapping of Banks")
plt.xlim(0,110)
plt.ylim(0,80)
legend1 = plt.legend(scatter_plot, banks2.keys(), loc="upper left")
plt.gca().add_artist(legend1)  # Add the color-based legend to the plot without removing the scatter plot legend


		

plt.show()
plt.close()



## To plot it as a colormap
#bank_gridmap = []
#
#for bank in range(0,8):
#	bank_gridmap.append(np.zeros((80, 100)))
#
#for bank in banks2:
#	bank_id = banks_list.index(bank)
#	for coor in banks2[bank]:
#		bank_gridmap[bank_id][coor[1]][coor[0]] =1
#	
## Plotting
#fig, ax = plt.subplots(figsize=(8, 10))
#
#im =[ ]
#colormaps = ['Purples', 'Blues', 'Greens', 'Reds', 'Oranges', 'YlOrBr', 'Greys', 'binary']
#
#for bank in range(0,8):
#	# Plot the first dimension with 'viridis' colormap
#	im.append(ax.imshow(bank_gridmap[bank], cmap=colormaps[bank], interpolation='nearest', alpha=1))
#
##cbar = []
##for bank in range(0,8):
##	cbar.append(fig.colorbar(im[bank], ax=ax, label=str(banks_list[bank])))
### Set color legends
##for bank in range(0,8):
##	cbar[bank].set_ticks([0, 0.5, 1])
##	cbar[bank].set_ticklabels(['NP', '', 'P'])
#
## Add titles and labels
#plt.title('Bank Colormap')
#plt.xlabel('X-axis')
#plt.ylabel('Y-axis')
#
#plt.show()
#plt.close()

bank_overlap = np.zeros((80, 100))
for bank in banks2:
	bank_id = banks_list.index(bank)
	for coor in banks2[bank]:
		bank_overlap[coor[1]][coor[0]] +=0.125


plt.imshow(bank_overlap, cmap='Greys', interpolation='nearest', alpha=0.125)
plt.title('Bank Overlap')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

plt.show()
plt.close()


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






#print("row_correlation_x_data")
#print(row_correlation_x_data)
#print("row_correlation_y_data")
#print(row_correlation_y_data)
#print("col_correlation_x_data")
#print(col_correlation_x_data)
#print("col_correlation_y_data")
#print(col_correlation_y_data)
#print("bit_correlation_x_data")
#print(bit_correlation_x_data)
#print("bit_correlation_y_data")
#print(bit_correlation_y_data)


#######################################################################################################################################################################################
######################                                                      XY Correlation for Rows and Columns								###############
#######################################################################################################################################################################################

print("XY_Row_Correlation")
for key in rows2:
	if len(rows2[key])>1:
		print(key,rows2[key])


print("XY_Row_Correlation spread")
rows2_spread = {}
rows2_spread_max = [-300, -300]

for key in rows2:
	if len(rows2[key])>1:
		X_MAX = -300
		X_MIN = 300
		Y_MAX = -300
		Y_MIN = 300
		for coors in rows2[key]:
			if coors[0] > X_MAX:
				X_MAX = coors[0]
			if coors[0] < X_MIN:
				X_MIN  = coors[0]
			if coors[1] > Y_MAX:
				Y_MAX = coors[1]
			if coors[1] < Y_MIN:
				Y_MIN  = coors[1]
		rows2_spread[key] = [X_MAX - X_MIN, Y_MAX - Y_MIN ] 
print("Spread of different rows")			
for key in rows2_spread:
	print(key, rows2_spread[key])
	if rows2_spread[key][0] > rows2_spread_max[0]:
		rows2_spread_max[0] = rows2_spread[key][0]
	if rows2_spread[key][1] > rows2_spread_max[1]:
		rows2_spread_max[1] = rows2_spread[key][1]

print("Maximum spread of a row")
print(rows2_spread_max)


print("XY_Column_Correlation")
for key in cols2:
	if len(cols2[key])>1:
		print(key,cols2[key])





