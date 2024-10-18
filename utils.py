import os
import csv

def append_to_csv_file(dict_data, file_path, fieldnames):
            write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
            with open(file_path, 'a', newline='') as file:
                
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader() if write_header else None
                # No need to write the header each time; just append rows
                writer.writerow(dict_data)
                print("Metrics data saved to", file_path)
                
                
# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# def draw_fig_from_csv(file_column_dict, fig_name, located, save_path):
    
#     if not isinstance(file_column_dict, dict):
#         raise ValueError('file_column_dict must be a dictionary')
#     if located is not None:
#         for column, value in file_column_dict.items():
            
    
#     for file, column in file_column_dict.items():
#         df = pd.read_csv(file)
#         plt.figure(figsize=(10, 6))
#         plt.plot(df[column], label=column)
#         plt.title(fig_name , ' Graph of ' + column)
#         plt.xlabel('Last Epoch')
#         plt.ylabel(column)
#         plt.legend()  # Show legend
#         plt.grid(True)  # Show grid
#         pth = os.path.join(save_path, fig_name, column + '.png')
#         plt.savefig('your_graph_filename.png', dpi=300)  # Saves the figure to a file. You can change the format to PDF, SVG, etc., by changing the file extension.
#         print('Graph saved at:', pth)