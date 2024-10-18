import csv
import os



def append_to_csv_file(dict_data, file_path):
    write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
    # if write_header:
    #     with open(file_path, 'wb'):
    #         wr = csv.writer(file_path)
    with open(file_path, 'a', newline='') as file:
        
        writer = csv.DictWriter(file, fieldnames=dict_data.keys())
        writer.writeheader() if write_header else None
        # No need to write the header each time; just append rows
        writer.writerow(dict_data)
        print("Metrics data saved to", file_path)