import sys
import laser_lib as ll
import pandas as pd

### DDR4 Parameter, set according to need.
if((len(sys.argv)>3) and (sys.argv[3]=="DDR4")):
    DDR4 =True
else:
    DDR4 = False

### calculate errors for each bank per line
def calc_bank_errors(bank_row_dict, bank):
    errors = 0
    for row in bank_row_dict[bank]:
        for col in bank_row_dict[bank][row]:
            for bit in bank_row_dict[bank][row][col]:
                errors += bank_row_dict[bank][row][col][bit][0] + bank_row_dict[bank][row][col][bit][1]
    return errors

lines = ll.parse_run_directory(sys.argv[1], sys.argv[2])
lines = sorted(lines, key=lambda line: line.name)

data = []
bank_error_data = {}

### Collect all unique bank numbers
all_banks = set()
for line in lines:
    all_banks.update(line.bank_row_dict.keys())

### Create column names for banks, with a blank column between Total Error and banks
columns = ['Coordinate=Value', 'Total Error', ''] + [f"Bank {bank}" for bank in sorted(all_banks)]

for line in lines:
    row_base = [f"{line.coor} = {line.value}"]
    total_error = sum(calc_bank_errors(line.bank_row_dict, bank) for bank in line.bank_row_dict)
    row = row_base + [total_error, '']  # Adding a blank cell
    
    bank_errors = {f"Bank {bank}": calc_bank_errors(line.bank_row_dict, bank) for bank in line.bank_row_dict}
    
    # Fill in bank error columns
    for bank in sorted(all_banks):
        row.append(bank_errors.get(f"Bank {bank}", 0))
    
    data.append(row)

    for bank in line.bank_row_dict:
        error = calc_bank_errors(line.bank_row_dict, bank)
        if bank not in bank_error_data:
            bank_error_data[bank] = error
        else:
            bank_error_data[bank] += error

### Create DataFrame for line-specific data
df = pd.DataFrame(data, columns=columns)
df.to_excel('output_per_line.xlsx', index=False)

### Create DataFrame for total error per bank
df_bank = pd.DataFrame(list(bank_error_data.items()), columns=['Bank', 'Total Error per Bank'])
df_bank.to_excel('output_per_bank.xlsx', index=False)


for line in lines:
    if line.coor == "X":
        bank_set = set(line.bank_row_dict.keys())
        print(f"Line {line.coor} = {line.value} has banks: {bank_set}")
