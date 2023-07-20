trades = []

import csv

space = '	'
with open('/Users/motin/Downloads/traffic/traffic/XAUUSD15.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        for key, value in row.items():
            keys_list = key.split(space)
            values_list = value.split(space)

            trades.append({
                'Time': keys_list[0],
                'Open': keys_list[1],
                'High': keys_list[2],
                'Low': keys_list[3],
                'Close': keys_list[4],
                'Timeframe': keys_list[5],
            })

            trades.append({
                'Time': values_list[0],
                'Open': values_list[1],
                'High': values_list[2],
                'Low': values_list[3],
                'Close': values_list[4],
                'Timeframe': values_list[5],
            })

with open('/Users/motin/Downloads/traffic/traffic/XAUUSD15.csv', 'w') as files:
    writer = csv.DictWriter(files,  fieldnames=['Time', 'Open', 'High', 'Low', 'Close', 'Timeframe'])
    writer.writeheader()
    writer.writerows(trades)    

